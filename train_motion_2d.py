#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import time
import yaml
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter as Logger

from utility.dataset.kitti.parser_multiscan import Parser
from modules.model_motion_2d import MotionNet
from modules.lonet import OdomRegNet
from modules.psmnet import PSMNet
from modules.nets.deeplio import DeepLO
from utility.warmupLR import *


parser = argparse.ArgumentParser(description='Training')
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
    help='Directory to put the log data. Default: ~/logs/date+time')
parser.add_argument(
    '--pretrained', '-p',
    type=str,
    required=False,
    default=None,
    help='Directory to get the pretrained model. If not passed, do from scratch!')


def main():
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.log is None and FLAGS.pretrained is not None:
        FLAGS.log = FLAGS.pretrained
    if FLAGS.log is None and FLAGS.pretrained is None:
        FLAGS.log = 'output/logs/' + datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M")
         
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
    print("pretrained", FLAGS.pretrained)
    print("----------\n")
    
    # start training
    main_worker(FLAGS)
    
def main_worker(args):
    # open arch config file
    try:
        print("Opening arch config file %s" % args.arch_cfg)
        ARCH = yaml.safe_load(open(args.arch_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()
    # open data config file
    try:
        print("Opening data config file %s" % args.data_cfg)
        DATA = yaml.safe_load(open(args.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()
     
    start_epoch = 0
    max_epoch = ARCH["train"]["max_epochs"]
    tb_logger = Logger(args.log + "/tb")
    
    # Data loading code
    parser = Parser(root=args.dataset,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
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
    train_loader = parser.get_train_set()
    
    # create model
    print("=> creating model '{}'".format(args.arch_cfg))
    with torch.no_grad():
        # model = MotionNet()
        model = DeepLO((3,64,2048))
        # model = OdomRegNet(5)
        # model = PSMNet(192)
    # model.apply(weights_init)
    
    if torch.cuda.is_available():
        model.cuda()
        cudnn.benchmark = True
        cudnn.fastest = True
    if ARCH["train"]["optimizer"] == "SDG":
        optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                    lr=ARCH["train"]["lr"],
                                    momentum=ARCH["train"]["momentum"],
                                    weight_decay=ARCH["train"]["w_decay"])
    elif ARCH["train"]["optimizer"] == "Adam":
        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                    lr=ARCH["train"]["lr"],
                                    weight_decay=ARCH["train"]["w_decay"])
    else:
        raise NotImplementedError
    
    # Use warmup learning rate
    # post decay and step sizes come in epochs and we want it in steps
    steps_per_epoch = parser.get_train_size()
    up_steps = int(ARCH["train"]["wup_epochs"] * steps_per_epoch)
    final_decay = ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
    # scheduler = warmupLR(optimizer=optimizer,
    #                     lr=ARCH["train"]["lr"],
    #                     warmup_steps=up_steps,
    #                     momentum=ARCH["train"]["momentum"],
    #                     decay=final_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.pretrained is not None:
        model_path = args.pretrained + "/Mos_odom.pth.tar"
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found in '{}'".format(args.pretrained))
            model.apply(init_weights)
    
    cudnn.benchmark = True
    
    # save train info
    best_train_iou = 0
    best_valid_iou = 0
    for epoch in range(start_epoch, max_epoch):
        # train for one epoch
        train_iou = train_epoch(train_loader, model, optimizer, scheduler, epoch, max_epoch, tb_logger)
        # checkpoint save
        if train_iou > best_train_iou:
            best_train_iou = train_iou
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'motion',
                'state_dict': model.state_dict(),
            }, args.log, suffix='_train_best')
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'motion',
                'state_dict': model.state_dict(),
            }, args.log, suffix='')
            
        # if epoch % ARCH["train"]["report_epoch"] == 0:
        #     # evaluate on validation set
        #     print("*" * 70)
        #     valid_iou = validate(valid_loader, model, parser.get_xentropy_class_string, epoch, tb_logger)
        #     if valid_iou > best_valid_iou:
        #         best_valid_iou = valid_iou
        #         save_checkpoint({
        #             'epoch': epoch + 1,
        #             'arch': 'motion',
        #             'state_dict': model.state_dict(),
        #         }, args.log, suffix='_valid_best')
        print('Finished One Epoch Training and Save Ckpt!')
    print("*" * 80)
    print('Finished Training')
        
def train_epoch(train_loader, model, optimizer, scheduler,epoch, max_epoch, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Losses', ':.4f')
    losses_tran = AverageMeter('Loss_tran', ':.4f')
    losses_rot = AverageMeter('Loss_rot', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, losses_tran, losses_rot],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (in_vol, _, _, _, _, path_name, _, _, _, _, _, _, _, _, _,tran_list, rot_list) in tqdm(enumerate(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        in_vol = in_vol.cuda()
        tran_labels = tran_list[-1].cuda().float()
        rot_labels = rot_list[-1].cuda().float()
        
        # compute output and loss
        loss, tran, rot = model(in_vol,tran_labels, rot_labels)
        if i % 10 == 0:
            print('***********')
            print('output tran:',tran.data)
            print('labels tran:',tran_labels)
            print('output rot:',rot.data)
            print('labels rot:',rot_labels)
            print('loss:',loss)
        loss_sum = loss['sum']
        loss_tran = loss['tran']
        loss_rot = loss['rot']

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        
        losses.update(loss_sum.item(), in_vol.size(0))
        losses_tran.update(loss_tran.item(), in_vol.size(0))
        losses_rot.update(loss_rot.item(), in_vol.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            progress.display(i)
            print('train time left: ',calculate_estimate(max_epoch,epoch,i,len(train_loader),data_time.avg, batch_time.avg))
            
    # step scheduler
    scheduler.step()
            
    # tensorboard logger
    logger.add_scalar('train_loss_sum', losses.avg, epoch)
    logger.add_scalar('train_loss_tran', losses_tran.avg, epoch)
    logger.add_scalar('train_loss_rot', losses_rot.avg, epoch)

    return losses.avg

    
def validate(val_loader, model, class_func, epoch, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Losses', ':.4f')
    losses_tran = AverageMeter('Loss_tran', ':.4f')
    losses_rot = AverageMeter('Loss_rot', ':.4f')

    # switch to evaluate mode
    model.eval()
    # empty the cache to infer in high res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        end = time.time()
        for i, (in_vol, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _,tran_list, rot_list) in tqdm(enumerate(val_loader)):
            # measure data loading time
            data_time.update(time.time() - end)
            in_vol = in_vol.cuda()
            tran_labels = tran_list[-1].cuda().float()
            rot_labels = rot_list[-1].cuda().float()
            
            # compute output and loss
            loss, tran, rot = model(in_vol[:,-1],in_vol[:,-2],tran_labels, rot_labels)
            loss_sum = loss['sum']
            loss_tran = loss['tran']
            loss_rot = loss['rot']
            
            losses.update(loss_sum.item(), in_vol.size(0))
            losses_tran.update(loss_tran.item(), in_vol.size(0))
            losses_rot.update(loss_rot.item(), in_vol.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        
    print('Validation set:\n'
                'Time avg per batch {batch_time.avg:.3f}\n'
                'Loss avg {loss.avg:.4f}\n'
                .format(batch_time =batch_time,
                                    loss=losses,
                                    ))    
       
    # tensorboard logger
    logger.add_scalar('valid_loss_sum', losses.avg, epoch)
    logger.add_scalar('valid_loss_tran', losses_tran.avg, epoch)
    logger.add_scalar('valid_loss_rot', losses_rot.avg, epoch)
    
    return losses.avg

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def calculate_estimate(max_epoch, epoch, iter, len_data, data_time_t, batch_time_t):
        estimate = int((data_time_t + batch_time_t) * (len_data * max_epoch - (iter + 1 + epoch * len_data)))
        return str(datetime.timedelta(seconds=estimate))

def save_checkpoint(state, logdir, suffix=""):
    # Save the weights
    torch.save(state, logdir +
               "/Mos_odom" + suffix + '.pth.tar')

def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':   
    main()

        

    
