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
from modules.model_motion_2d import MotionNet
from modules.lonet import OdomRegNet
from modules.nets.deeplio import DeepLO
from utility.warmupLR import *
from utility.geometry import get_transformation_matrix_quaternion, get_transformation_matrix_euler
from utility.dataset.kitti.utils import write_poses, load_calib

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
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir, split):
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
                    split=self.split,
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=False)
    
    calib_file = os.path.join(self.datadir, "sequences",
                            "08",  "calib.txt")
    T_cam_velo = load_calib(calib_file)
    self.T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    self.T_velo_cam = np.linalg.inv(T_cam_velo)

    with torch.no_grad():
        # self.model = MotionNet()
        self.model = DeepLO((3,64,2048))
        # self.model = OdomRegNet(5)
        self.model = load_model(modeldir, self.model)

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
    # initial last_pose matrix
    last_pose = np.eye(4)
    with torch.no_grad():
      end = time.time()
      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints,tran_list, rot_list,) in tqdm(enumerate(loader)):
        # first cut to rela size (batch size one allows it)
        path_seq = path_seq[0]
        tran_labels = tran_list[-1].cuda().float()
        rot_labels = rot_list[-1].cuda().float()
        proj_in = proj_in.cuda()

        #compute output
        loss, tran, rot = self.model(proj_in,tran_labels, rot_labels)
        
        # save pose
        # relative_matrix = get_transformation_matrix_quaternion(tran, rot)
        relative_matrix = get_transformation_matrix_euler(tran, rot)
    
        pose_path = os.path.join(self.logdir, "sequences",
                            path_seq, "poses.txt")
        
        # write pose and update last_pose
        last_pose = write_poses(pose_path, np.dot(self.T_cam_velo,np.dot(relative_matrix,self.T_velo_cam)), last_pose)


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
    
    