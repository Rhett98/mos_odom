#!/usr/bin/env python3
import os
import sys
path = os.getcwd()
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.resnet3d import ResNet3D, BasicBlock, SEBasicBlock
from modules.resnet2p1d import ResNet2P1D, BasicBlock2p1d
from modules.BaseBlocks import ResBlockDP, UpBlockDP
from modules.losses import Lovasz_softmax, UncertaintyLoss


class MotionNet(nn.Module):
    def __init__(self, input_scan=3, motion_backbone = 'resnet3d'):
        """
        Ues 3DCNN to ectract feature from tensor[bsize,n_scans,c,h,w]
        """
        super(MotionNet, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='mean').float()
        self.uncertainty_loss = UncertaintyLoss(2)
        
        if motion_backbone == 'resnet3d':
            self.backbone = ResNet3D(BasicBlock, [1, 1, 1, 1],[32, 64, 128, 256],input_scan)
        elif motion_backbone == 'resnet2p1d':
            self.backbone = ResNet2P1D(BasicBlock2p1d, [1, 1, 1, 1],[32, 64, 128, 256],input_scan)
        elif motion_backbone == 'se-resnet3d':
            self.backbone = ResNet2P1D(SEBasicBlock, [1, 1, 1, 1],[32, 64, 128, 256],input_scan)
        else:
            raise Exception("Not define motion backbone correctly!")

        # FC layer to odom output
        rot_out_features = 4
        self.fully_connected_translation = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=300, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=3))
        self.fully_connected_rotation = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=300, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=rot_out_features))
        # print(sum(p.numel() for p in self.backbone.parameters())/1000000.0)
        # print(sum(p.numel() for p in self.fully_connected_translation.parameters())/1000000.0)
        # print(sum(p.numel() for p in self.fully_connected_rotation.parameters())/1000000.0)
        
    def forward(self, x, tran_labels, rot_labels):
        """Use resnet to extract feature and predict
        Args:
            images (Tensor): Rangeimage time seq, B*n*c*H*W

        Returns:
            feature_list:
               [[bsize, 32, 3, 16, 512],
                [bsize, 64, 2, 8, 256],
                [bsize, 128, 1, 4, 128],
                [bsize, 256, 1, 2, 64],
                [bsize, 256],
                [bsize, 300]]
            pose: [x,y,z,q0,q1,q2,q3]
        """
        feature_list = self.backbone(x)
        feature = feature_list[-1]
        rotation = self.fully_connected_rotation(feature)
        translation = self.fully_connected_translation(feature)
        rotation = rotation / torch.norm(rotation)
        
        loss = {}
        loss_tran = self.l2_loss(tran_labels, translation)
        loss_rot =  self.l2_loss(rot_labels, rotation/torch.norm(rotation))
        loss_sum = self.uncertainty_loss(loss_tran, loss_rot)

        loss['tran'] = loss_tran
        loss['rot'] = loss_rot
        loss['sum'] = loss_sum
        return loss, translation, rotation


class MosNet(nn.Module):
    def __init__(self, pretrain=None, weight_loss=None, freeze_sematic=True, motion_backbone = 'resnet3d'):
        super(MosNet,self).__init__()
        self.nclasses = 3
        # define loss function
        self.nll_loss = nn.NLLLoss(weight=weight_loss)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.Ls = Lovasz_softmax(ignore=0)
        self.uncertainty_loss = UncertaintyLoss(3)
        
        # define layer
        self.sematic = SematicNet(pretrain,freeze_base=freeze_sematic)
        self.motion = MotionNet(motion_backbone = motion_backbone)
        
        self.fusion_layer1 = ResBlockDP(224, 128, 0.2, pooling=False)
        self.fusion_layer2 = ResBlockDP(384, 256, 0.2, pooling=False)
        self.fusion_layer3 = ResBlockDP(384, 256, 0.2, pooling=False)
        
        self.resBlock1 = ResBlockDP(128, 256, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlockDP(256, 256, 0.2, pooling=True)
        self.resBlock3 = ResBlockDP(256, 256, 0.2, pooling=False)
        
        self.upBlock1 = UpBlockDP(256, 128, 0.2)
        self.upBlock2 = UpBlockDP(128, 128, 0.2)
        self.upBlock3 = UpBlockDP(128, 64, 0.2)
        self.upBlock4 = UpBlockDP(64, 32, 0.2, drop_out=False)
        
        self.logits = nn.Conv2d(32, self.nclasses, kernel_size=(1, 1))
        # print('******')
        # print(sum(p.numel() for p in self.sematic.parameters())/1000000.0)
        # print(sum(p.numel() for p in self.motion.parameters())/ 1000000.0)
        
    def forward(self, time_seq, seg_labels, tran_labels, rot_labels):
        """
        time_seq: shape [bsize, n, c, h, w]
        
        """
        ###### the Encoder for current image to extract sematic feature ######
        x = self.sematic(time_seq)  
        # x: [bsize, 256, 4, 128]
        
        ###### the Encoder for image sequences to extract motion feature ######
        y, translation, rotation = self.motion(time_seq) 
        # y:list[[],[],[]]; translation: [bsize, 3]; rotation: [bsize, 4]
        
        ###### fuse 2 specific branches ######
        # resdual from sematic-net
        down0b, down1b = x[0], x[1]
        # s-layer1:[bsize, 128, 16, 512] + m-layer0:[bsize, 32, 3, 16, 512]
        s1 = x[2]
        m1 = y[0].view(y[0].shape[0], y[0].shape[1]*y[0].shape[2], y[0].shape[3], y[0].shape[4])
        # [1, 224, 16, 512] -> [1, 32, 16, 512]
        fu1 = self.fusion_layer1(torch.cat([s1, m1],dim=1))   
        # s-layer2:[bsize, 256, 8, 256] + m-layer1:[bsize, 64, 2, 8, 256]
        s2 = x[3]
        m2 = y[1].view(y[1].shape[0], y[1].shape[1]*y[1].shape[2], y[1].shape[3], y[1].shape[4])
        # [1, 384, 8, 256] -> [1, 64, 8, 256]
        fu2 = self.fusion_layer2(torch.cat([s2, m2],dim=1))
        # s-layer4:[bsize, 256, 4, 128] + m-layer2:[bsize, 128, 1, 4, 128]
        s3 = x[4]
        m3 = y[2].view(y[2].shape[0], y[2].shape[1]*y[2].shape[2], y[2].shape[3], y[2].shape[4])
        # [1, 384, 4, 128] -> [1, 128, 4, 128]
        fu3 = self.fusion_layer3(torch.cat([s3, m3],dim=1))
        
        ##### fusion-net ######
        down2c, down2b = self.resBlock1(fu1)
        add1 = torch.add(down2c,fu2)
        down3c, down3b = self.resBlock2(add1)
        add2 = torch.add(down3c,fu3)
        down5c = self.resBlock3(add2)

        up4e = self.upBlock1(down5c,down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)

        logits = logits
        logits = F.softmax(logits, dim=1)
        
        ##### calculate multi-task loss #####
        loss = {}       
        loss_seg = self.nll_loss(torch.log(logits.clamp(min=1e-8)), seg_labels.long()) + self.Ls(logits, seg_labels.long())
        loss_tran = self.l1_loss(tran_labels, translation)
        loss_rot =  self.l1_loss(rot_labels, rotation/torch.norm(rotation))
        loss_sum = self.uncertainty_loss(loss_seg, loss_tran, loss_rot)
        
        # write loss to a dict
        loss['seg'] = loss_seg
        loss['tran'] = loss_tran
        loss['rot'] = loss_rot
        loss['sum'] = loss_sum
        return loss, logits, translation, rotation

if __name__ == '__main__':
    from thop import profile
    model = MosNet('pretrained/SalsaNextEncoder')
    dummy_input = torch.randn(1, 3, 5, 64, 2048),torch.zeros(1, 64, 2048),torch.randn(1,3),torch.randn(1,4)
    # model2 = SematicNet('pretrained/SalsaNextEncoder',1)
    # dummy_input2 = torch.randn(1, 3, 5, 64, 2048)
    # model3 = MotionNet()
    # dummy_input3 = torch.randn(1, 3, 5, 64, 2048)
    flops, params = profile(model, (dummy_input))
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # with SummaryWriter(comment='MosNet') as w1:
    #     w1.add_graph(model1, (dummy_input1))
    
    ###### save salsanext-encoder model
    # model = SalsaNextEncoder(20, 1)
    # checkpoint = torch.load('/home/yu/Resp/pretrained/pretrained/SalsaNext')
    # state_dict = model.state_dict()
    # pretrained_dict = checkpoint["state_dict"]
    # for key in state_dict:
    #     if 'module.'+key in pretrained_dict:
    #         state_dict[key] = pretrained_dict['module.'+key]
    #     # if key in pretrained_dict:
    #     #     state_dict[key] = pretrained_dict[key]
    #     else:
    #         print('checkpoint layer name is wrong!!!')
    # model.load_state_dict(state_dict, strict=True)
    # torch.save({'state_dict': model.state_dict()}, 'pretrained/SalsaNextEncoder')
    
    # x = torch.randn(1, 3, 5, 64, 2048)
    # c = x[:,-1]
    # print(c.shape)