import os
import sys
path = os.getcwd()
sys.path.append(path)
import time
import datetime
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from utility.KNN import KNN
from utility.dataset.kitti.parser_multiscan import Parser
from modules.model_infer import MosNet
from utility.warmupLR import *
from utility.dataset.kitti.utils import write_poses

def load_model(weight_path, model):
    state_dict = model.state_dict()

    ckpt = torch.load(weight_path)
    pretrained_dict = ckpt["state_dict"]

    for key in state_dict:
        if key in pretrained_dict:
            state_dict[key] = pretrained_dict[key]
        else:
            print(key)

    model.load_state_dict(state_dict, strict=True)
    return model

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.split = split

    # get the data
    self.parser = Parser(root=self.datadir,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=self.DATA["split"]["test"],
                    split='train',
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=True)
    # # weights for loss (and bias)
    # epsilon_w = ARCH["train"]["epsilon_w"]
    # content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
    # for cl, freq in DATA["content"].items():
    #     x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
    #     content[x_cl] += freq
    # loss_w = 1 / (content + epsilon_w)  # get weights
    # for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
    #     if DATA["learning_ignore"][x_cl]:
    #         # don't weigh
    #         loss_w[x_cl] = 0
    # print("Loss weights from content: ", loss_w.data)
    
    # ignore_class = []
    # for i, w in enumerate(loss_w):
    #     if w < 1e-10:
    #         ignore_class.append(i)
    #         print("Ignoring class ", i, " in IoU evaluation")
    # concatenate the encoder and the head
    with torch.no_grad():
        self.model = MosNet(
                       freeze_sematic=ARCH["train"]["freeze_sematic"],
                       motion_backbone=ARCH["train"]["motion_backbone"])
        # w_dict = torch.load(modeldir, map_location=lambda storage, loc: storage)
        self.model = load_model(modeldir, self.model)
        # self.model.load_state_dict(w_dict['state_dict'], strict=True)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    cnn = []
    knn = []
    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn,cnn,knn):
    # switch to evaluate mode
    self.model.eval()
    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints,_, _,) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]
        
        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        #compute output
        proj_output, tran, rot = self.model(proj_in)
        proj_argmax = proj_output[0].argmax(dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
                "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
                "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if self.post:
            # knn postproc
            unproj_argmax = self.post(proj_range,
                                        unproj_range,
                                        proj_argmax,
                                        p_x,
                                        p_y)
        else:
            # put in original pointcloud using indexes
            unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("KNN Infered seq", path_seq, "scan", path_name,
                "in", res, "sec")
        knn.append(res)
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)
        
        # save pose
        path_pose = os.path.join(self.logdir, "sequences",
                            path_seq, "poses.txt")
        write_poses(path_pose, tran, rot)



if __name__ == '__main__': 
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',)
    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        required=True,
        help='Architecture yaml cfg file. See /config/arch for sample. No default!',)
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default='config/labels/semantic-kitti-mos.yaml',
        help='Classification yaml cfg file. See /config/labels for sample. No default!',)
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=None,
        help='Directory to output the infer data. Default: ~/output')
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the model!')  
    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default='valid',
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',)
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.log is None :
        FLAGS.log = 'output/' + datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M")
         
    # open arch config file
    try:
        print("Opening arch config file %s" % FLAGS.arch_cfg)
        ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()
    
    # print summary of what we will do
    print("----------")
    print("dataset", FLAGS.dataset)
    print("arch_cfg", FLAGS.arch_cfg)
    print("data_cfg", FLAGS.data_cfg)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("----------\n")
    
    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        for seq in DATA["split"]["train"]:
            seq = '{0:02d}'.format(int(seq))
            print("train", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["valid"]:
            seq = '{0:02d}'.format(int(seq))
            print("valid", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
        for seq in DATA["split"]["test"]:
            seq = '{0:02d}'.format(int(seq))
            print("test", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise
    
    # create user and infer dataset
    user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model,FLAGS.split)
    user.infer()
    
    