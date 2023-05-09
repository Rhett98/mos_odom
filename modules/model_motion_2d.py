#!/usr/bin/env python3
import os
import sys
path = os.getcwd()
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.resnet import BasicBlock, ResNet
from modules.losses import UncertaintyLoss


class MotionNet(nn.Module):
    def __init__(self):
        """
        Ues 3DCNN to ectract feature from tensor[bsize,n_scans,c,h,w]
        """
        super(MotionNet, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='mean').float()
        self.uncertainty_loss = UncertaintyLoss(2)
        
        self.backbone = ResNet(BasicBlock, (2, 2, 2, 2))
        self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
        # FC layer to odom output
        rot_out_features = 4
        self.fc = nn.Linear(1024,400)
        self.fully_connected_translation = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=400, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=3))
        self.fully_connected_rotation = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=400, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=rot_out_features))
        
    def forward(self, x1, x2, tran_labels, rot_labels):
        """Use resnet to extract feature and predict
        Args:
            images (Tensor): Rangeimage time seq, B*n*c*H*W

        Returns:
            loss
            pose: [x,y,z,q0,q1,q2,q3]
        """
        feature_1 = self.backbone(x1)
        feature_2 = self.backbone(x2)
        feature = torch.cat([feature_1, feature_2],dim=1)
        
        f = self.avgpool(feature)
        f = f.view(f.size(0), -1)
        f = self.fc(f)
        rotation = self.fully_connected_rotation(f)
        translation = self.fully_connected_translation(f)
        
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
    dummy_input = torch.randn(1, 5, 64, 2048),torch.randn(1, 5, 64, 2048),torch.randn(1, 3),torch.randn(1, 4),
    flops, params = profile(model, (dummy_input))
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))