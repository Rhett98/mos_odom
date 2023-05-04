#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import time
import yaml
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter as Logger

from utility.dataset.kitti.parser_multiscan import Parser
from utility.ioueval import iouEval
from modules.model import MosNet
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
    
    # weights for loss (and bias)
    epsilon_w = ARCH["train"]["epsilon_w"]
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
    
    ignore_class = []
    for i, w in enumerate(loss_w):
        if w < 1e-10:
            ignore_class.append(i)
            print("Ignoring class ", i, " in IoU evaluation")
    
    # create model
    print("=> creating model '{}'".format(args.arch_cfg))
    with torch.no_grad():
        model = MosNet(ARCH["train"]["salsanext_path"],weight_loss=loss_w, 
                       freeze_sematic=ARCH["train"]["freeze_sematic"],
                       motion_backbone=ARCH["train"]["motion_backbone"])
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
    scheduler = warmupLR(optimizer=optimizer,
                        lr=ARCH["train"]["lr"],
                        warmup_steps=up_steps,
                        momentum=ARCH["train"]["momentum"],
                        decay=final_decay)
    
    # optionally resume from a checkpoint
    if args.pretrained is not None:
        model_path = args.pretrained + "/Mos_odom.pth.tar"
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found in '{}'".format(args.pretrained))
    
    # set train and valid evaluator
    evaluator = iouEval(parser.get_n_classes(), device, ignore_class)
    
    cudnn.benchmark = True
    
    # save train info
    best_train_iou = 0
    best_valid_iou = 0
    for epoch in range(start_epoch, max_epoch):
        # train for one epoch
        train_iou = train_epoch(train_loader, model, optimizer, evaluator, scheduler, epoch, max_epoch, tb_logger)
        
        # checkpoint save
        if train_iou > best_train_iou:
            best_train_iou = train_iou
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'mos',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, args.log, suffix='_train_best')
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'mos',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, args.log, suffix='')
            
        if epoch % ARCH["train"]["report_epoch"] == 0:
            # evaluate on validation set
            print("*" * 70)
            valid_iou = validate(valid_loader, model, evaluator, parser.get_xentropy_class_string, epoch, tb_logger)
            if valid_iou > best_valid_iou:
                best_valid_iou = valid_iou
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'mos',
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, args.log, suffix='_valid_best')
    print("*" * 80)
    print('Finished Training')
        
def train_epoch(train_loader, model, optimizer, evaluator, scheduler,epoch, max_epoch, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Losses', ':.4f')
    losses_seg = AverageMeter('Loss_seg', ':.4f')
    losses_tran = AverageMeter('Loss_tran', ':.4f')
    losses_rot = AverageMeter('Loss_rot', ':.4f')
    iou = AverageMeter('Iou', ':.4f')
    acc = AverageMeter('Acc', ':.4f')
    iou_moving = AverageMeter('Iou_moving', ':.4f')
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
        in_vol = in_vol.cuda()
        proj_labels = proj_labels.cuda()
        tran_labels = tran_list[-2].cuda()
        rot_labels = rot_list[-2].cuda()
        
        # compute output and loss
        loss, output, tran, rot = model(in_vol, proj_labels, tran_labels, rot_labels)
        
        loss_sum = loss['sum']
        loss_seg = loss['seg']
        loss_tran = loss['tran']
        loss_rot = loss['rot']
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_sum.backward()
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
        iou_moving.update(class_jaccard[-1].item(), in_vol.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 40 == 0:
            progress.display(i)
            print('train time left: ',calculate_estimate(max_epoch,epoch,i,len(train_loader),data_time.avg, batch_time.avg))
            
    # step scheduler
    scheduler.step()
            
    # tensorboard logger
    logger.add_scalar('train_loss_sum', losses.avg, epoch)
    logger.add_scalar('train_loss_seg', losses_seg.avg, epoch)
    logger.add_scalar('train_loss_tran', losses_tran.avg, epoch)
    logger.add_scalar('train_loss_rot', losses_rot.avg, epoch)
    logger.add_scalar('train_acc', acc.avg, epoch)
    logger.add_scalar('train_iou', iou.avg, epoch)
    
    return iou.avg

    
def validate(val_loader, model, evaluator, class_func, epoch, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Losses', ':.4f')
    losses_seg = AverageMeter('Loss_seg', ':.4f')
    losses_tran = AverageMeter('Loss_tran', ':.4f')
    losses_rot = AverageMeter('Loss_rot', ':.4f')
    iou = AverageMeter('Iou', ':.4f')
    iou_moving = AverageMeter('Iou_moving', ':.4f')
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
            loss, output, _, _ = model(in_vol, proj_labels, tran_labels, rot_labels)
            
            loss_sum = loss['sum']
            loss_seg = loss['seg']
            loss_tran = loss['tran']
            loss_rot = loss['rot']
            
            # measure accuracy and record loss
            argmax = output.argmax(dim=1)
            evaluator.addBatch(argmax, proj_labels)
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
                
            losses.update(loss_sum.item(), in_vol.size(0))
            losses_seg.update(loss_seg.item(), in_vol.size(0))
            losses_tran.update(loss_tran.item(), in_vol.size(0))
            losses_rot.update(loss_rot.item(), in_vol.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), in_vol.size(0))
        iou.update(jaccard.item(), in_vol.size(0))
        iou_moving.update(class_jaccard[-1].item(), in_vol.size(0))
        
    print('Validation set:\n'
                'Time avg per batch {batch_time.avg:.3f}\n'
                'Loss avg {loss.avg:.4f}\n'
                'Acc avg {acc.avg:.3f}\n'
                'mIoU avg {iou.avg:.3f}\n'
                'Iou moving avg {iou_moving.avg:.3f}\n'
                .format(batch_time =batch_time,
                                    loss=losses,
                                    acc=acc,
                                    iou=iou,
                                    iou_moving=iou_moving
                                    ))    
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=class_func(i), jacc=jacc))
       
    # tensorboard logger
    logger.add_scalar('valid_loss_sum', losses.avg, epoch)
    logger.add_scalar('valid_loss_seg', losses_seg.avg, epoch)
    logger.add_scalar('valid_loss_tran', losses_tran.avg, epoch)
    logger.add_scalar('valid_loss_rot', losses_rot.avg, epoch)
    logger.add_scalar('valid_acc', acc.avg, epoch)
    logger.add_scalar('valid_iou', iou.avg, epoch)
    logger.add_scalar('valid_iou_moving', iou_moving.avg, epoch)
    
    return iou.avg
    

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

    
