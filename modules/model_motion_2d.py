#!/usr/bin/env python3
import os
import sys
path = os.getcwd()
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.resnet import BasicBlock, ResNet
from modules.losses import HWSLoss


class MotionNet(nn.Module):
    def __init__(self):
        """
        Ues CNN to ectract feature from tensor[bsize,c,h,w]
        """
        super(MotionNet, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean').float()
        self.l2_loss = nn.MSELoss(reduction='mean').float()
        self.hws_loss = HWSLoss()
        
        self.backbone = ResNet(BasicBlock, (3, 4, 6, 3))
        self.fusion = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1,stride=2),
                                    nn.ReLU(),
                                    )
        self.avgpool =  nn.AdaptiveAvgPool2d((1, 4))
        # FC layer to odom output
        rot_out_features = 3
        self.fully_connected_translation = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4096, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=3))
        self.fully_connected_rotation = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4096, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=rot_out_features))
        
    def forward(self, x1, x2, tran_labels, rot_labels):
        """Use resnet to extract feature and predict
        Args:
            images (Tensor): Range image, 

        Returns:
            loss
            pose: [x,y,z,q0,q1,q2,q3]
        """
        feature_old = self.backbone(x1)
        feature_cur = self.backbone(x2)
        feature = torch.cat([feature_old, feature_cur],dim=1)
        feature = self.fusion(feature)
        feature = self.avgpool(feature)
        mov_f = feature.view(feature.size(0), -1)
        translation = self.fully_connected_translation(mov_f)
        rotation = self.fully_connected_rotation(mov_f)
        # rotation = rotation/torch.norm(rotation,dim=1).unsqueeze(1) 

        loss = {}
        loss_tran = self.l1_loss(tran_labels, translation)
        loss_rot =  self.l2_loss(rot_labels, rotation)
        loss_sum = self.hws_loss(loss_tran, loss_rot)

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

    