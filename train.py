#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import math
import os
import random
import shutil
import time
import yaml
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter as Logger

from utility.dataset.kitti.parser_multiscan import Parser
from utility.ioueval import iouEval
from modules.model import MosNet


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
    default="output",
    help='Directory to put the log data. Default: ~/logs/date+time')
parser.add_argument(
    '--pretrained', '-p',
    type=str,
    required=False,
    default=None,
    help='Directory to get the pretrained model. If not passed, do from scratch!')

def main():
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.log = FLAGS.log + '/logs/' + datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M")
    
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
    tb_logger = Logger(args.log)
    device = 'cuda'
    
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
    valid_loader = parser.get_valid_set()
    
    # create model
    print("=> creating model '{}'".format(args.arch_cfg))
    model = MosNet(3,'pretrained/SalsaNextEncoder')
    model.cuda()
    
    # infer learning rate before changing batch size
    init_lr = ARCH["train"]["lr"] * ARCH["train"]["batch_size"] / 256
    epsilon_w = ARCH["train"]["epsilon_w"]
    
    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=ARCH["train"]["momentum"],
                                weight_decay=ARCH["train"]["w_decay"])
    
    # optionally resume from a checkpoint
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    
    content = torch.zeros(parser.get_n_classes(), dtype=torch.float)
    for cl, freq in DATA["content"].items():
        x_cl = parser.to_xentropy(cl)  # map actual class to xentropy class
        content[x_cl] += freq
    loss_w = 1 / (content + epsilon_w)  # get weights
    for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
        if DATA["learning_ignore"][x_cl]:
            # don't weigh
            loss_w[x_cl] = 0
    print("Loss weights from content: ", loss_w.data)
    
    # set train and valid evaluator
    ignore_class = []
    for i, w in enumerate(loss_w):
        if w < 1e-10:
            ignore_class.append(i)
            print("Ignoring class ", i, " in IoU evaluation")
    evaluator = iouEval(parser.get_n_classes(), device, ignore_class)
    
    cudnn.benchmark = True
    
    # save train info
    best_train_iou = 0
    best_valid_iou = 0
    for epoch in range(start_epoch, max_epoch):
        adjust_learning_rate(optimizer, init_lr, epoch, max_epoch)
        # train for one epoch
        train_iou = train_epoch(train_loader, model, optimizer, evaluator, epoch, max_epoch, tb_logger)
        
        # checkpoint save
        if train_iou > best_train_iou:
            best_train_iou = train_iou
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'mos',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, args.log, is_best=True, filename='/checkpoint_{:04d}.pth.tar'.format(epoch))
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'mos',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, args.log, is_best=False, filename='/checkpoint_new.pth.tar')
            
        if epoch % ARCH["train"]["report_epoch"] == 0:
            # evaluate on validation set
            print("*" * 70)
            valid_iou = validate(valid_loader, model, evaluator, epoch, tb_logger)
            if valid_iou > best_valid_iou:
                best_valid_iou = valid_iou
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'mos',
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, args.log, is_best=True, is_valid=True, filename='/checkpoint_{:04d}.pth.tar'.format(epoch))
    print("*" * 80)
    print("Train is finished!")
        
def train_epoch(train_loader, model, optimizer, evaluator, epoch, max_epoch, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Losses', ':.4f')
    losses_seg = AverageMeter('Loss_seg', ':.4f')
    losses_tran = AverageMeter('Loss_tran', ':.4f')
    losses_rot = AverageMeter('Loss_rot', ':.4f')
    iou = AverageMeter('Iou', ':.4f')
    acc = AverageMeter('Acc', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, iou],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (in_vol, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _,tran_list, rot_list) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        in_vol = torch.split(in_vol, 10, dim=1)
        in_vol = in_vol.cuda()
        proj_labels = proj_labels.cuda()
        tran_labels = tran_list[-1].cuda()
        rot_labels = rot_list[-1].cuda()
        
        # compute output and loss
        loss, output, tran, rot = model(in_vol, proj_labels, tran_labels, rot_labels)
        
        loss_sum = loss['sum']
        loss_seg = loss['seg']
        loss_tran = loss['tran']
        loss_rot = loss['rot']
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            evaluator.reset()
            argmax = output.argmax(dim=1)
            evaluator.addBatch(argmax, proj_labels)
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            
        losses.update(loss_sum.item(), in_vol.size(0))
        losses_seg.update(loss_seg.item(), in_vol.size(0))
        losses_tran.update(loss_tran.item(), in_vol.size(0))
        losses_rot.update(loss_rot.item(), in_vol.size(0))
        acc.update(accuracy.item(), in_vol.size(0))
        iou.update(jaccard.item(), in_vol.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            progress.display(i)
            print('train time left: ',calculate_estimate(max_epoch,epoch,i,len(train_loader),data_time.avg, batch_time.avg))
            
    # tensorboard logger
    logger.add_scalar('train_loss_sum', losses.avg, epoch)
    logger.add_scalar('train_loss_seg', losses_seg.avg, epoch)
    logger.add_scalar('train_loss_tran', losses_tran.avg, epoch)
    logger.add_scalar('train_loss_rot', losses_rot.avg, epoch)
    logger.add_scalar('train_acc', acc.avg, epoch)
    logger.add_scalar('train_iou', iou.avg, epoch)
    
    return iou.avg

    
def validate(val_loader, model, evaluator, epoch, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Losses', ':.4f')
    losses_seg = AverageMeter('Loss_seg', ':.4f')
    losses_tran = AverageMeter('Loss_tran', ':.4f')
    losses_rot = AverageMeter('Loss_rot', ':.4f')
    iou = AverageMeter('Iou', ':.4f')
    acc = AverageMeter('Acc', ':.4f')

    # switch to evaluate mode
    model.eval()
    evaluator.reset()
    # empty the cache to infer in high res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    with torch.no_grad():
        end = time.time()
        for i, (in_vol, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _,tran_list, rot_list) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            in_vol = in_vol.cuda()
            proj_labels = proj_labels.cuda()
            tran_labels = tran_list[-1].cuda()
            rot_labels = rot_list[-1].cuda()
            
            # compute output and loss
            loss, output, tran, rot = model(in_vol, proj_labels, tran_labels, rot_labels)
            
            loss_sum = loss['sum']
            loss_seg = loss['seg']
            loss_tran = loss['tran']
            loss_rot = loss['rot']
            
            evaluator.reset()
            argmax = output.argmax(dim=1)
            evaluator.addBatch(argmax, proj_labels)
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
                
            losses.update(loss_sum.item(), in_vol.size(0))
            losses_seg.update(loss_seg.item(), in_vol.size(0))
            losses_tran.update(loss_tran.item(), in_vol.size(0))
            losses_rot.update(loss_rot.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
    # tensorboard logger
    logger.add_scalar('valid_loss_sum', losses.avg, epoch)
    logger.add_scalar('valid_loss_seg', losses_seg.avg, epoch)
    logger.add_scalar('valid_loss_tran', losses_tran.avg, epoch)
    logger.add_scalar('valid_loss_rot', losses_rot.avg, epoch)
    logger.add_scalar('valid_acc', acc.avg, epoch)
    logger.add_scalar('valid_iou', iou.avg, epoch)
    
    print('Validation set:\n'
                'Time avg per batch {batch_time.avg:.3f}\n'
                'Loss avg {loss.avg:.4f}\n'
                'Acc avg {acc.avg:.3f}\n'
                'IoU avg {iou.avg:.3f}'.format(batch_time=batch_time,
                                                loss=losses,
                                                acc=acc,
                                                iou=iou))
    return iou.avg
    

def calculate_estimate(max_epoch, epoch, iter, len_data, data_time_t, batch_time_t):
        estimate = int((data_time_t + batch_time_t) * (len_data * max_epoch - (iter + 1 + epoch * len_data)))
        return str(datetime.timedelta(seconds=estimate))

def save_checkpoint(state, log_path, is_best, is_valid=False, filename='/checkpoint.pth.tar'):
    path = log_path+filename
    torch.save(state, path)
    if is_best:
        if is_valid:
            shutil.copyfile(path, log_path+'/model_valid_best.pth.tar')
        else:
            shutil.copyfile(path, log_path+'/model_train_best.pth.tar')

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

def adjust_learning_rate(optimizer, init_lr, epoch, max_epoch):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

if __name__ == '__main__':   
    main()

    
