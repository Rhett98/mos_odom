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
            self.backbone = ResNet3D(BasicBlock, [2, 2, 2, 2],[64, 128, 256, 512],input_scan, n_classes=800)
        elif motion_backbone == 'resnet2p1d':
            self.backbone = ResNet2P1D(BasicBlock2p1d, [2, 2, 2, 2],[64, 128, 256, 512],input_scan, n_classes=800)
        elif motion_backbone == 'se-resnet3d':
            self.backbone = ResNet3D(SEBasicBlock, [2, 2, 2, 2],[64, 128, 256, 512],input_scan, n_classes=800)
        else:
            raise Exception("Not define motion backbone correctly!")

        # FC layer to odom output
        rot_out_features = 4
        self.fully_connected_translation = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=800, out_features=400),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=400, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=3))
        self.fully_connected_rotation = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=800, out_features=400),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=400, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=rot_out_features))
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
        
        loss = {}
        loss_tran = self.l2_loss(tran_labels, translation)
        loss_rot =  self.l2_loss(rot_labels, rotation/torch.norm(rotation))
        loss_sum = self.uncertainty_loss(loss_tran, loss_rot)

        loss['tran'] = loss_tran
        loss['rot'] = loss_rot
        loss['sum'] = loss_sum
        return loss, translation, rotation


if __name__ == '__main__':
    from thop import profile
    model = MotionNet()
    dummy_input = torch.randn(1, 3, 5, 64, 2048),torch.randn(1, 3),torch.randn(1, 4),
    flops, params = profile(model, (dummy_input))
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))