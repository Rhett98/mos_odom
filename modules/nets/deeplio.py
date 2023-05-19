import os
import sys
path = os.getcwd()
sys.path.append(path)
import torch
from torch import nn
from modules.nets.lidar_feat_nets import LidarPointSegFeat, LidarFlowNetFeat, LidarResNetFeat, LidarSimpleFeat1
from modules.nets.odom_feat_nets import OdomFeatFC
from modules.losses import HWSLoss

class DeepLO(nn.Module):
    """Base class for all DepplioN Networks"""
    def __init__(self, input_shape):
        super(DeepLO, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='mean').float()
        self.hws_loss = HWSLoss()
        self.p = 0.25
        self.input_shape = input_shape

        # self.lidar_feat_net = LidarPointSegFeat(self.input_shape)
        # self.lidar_feat_net = LidarFlowNetFeat(self.input_shape)
        self.lidar_feat_net = LidarResNetFeat(self.input_shape)
        # self.lidar_feat_net = LidarSimpleFeat1(self.input_shape)
        self.odom_feat_net = OdomFeatFC(128)

        if self.p > 0:
            self.drop = nn.Dropout(self.p)
        self.fc_pos = nn.Linear(256, 3)
        self.fc_ori = nn.Linear(256, 3)

    def forward(self, x, tran_labels, rot_labels):
        feat = self.lidar_feat_net(x)
        feat = self.odom_feat_net(feat)
        #b, s = x_last_feat.shape[0:2]
        #x_last_feat = x_last_feat.reshape(b*s, -1)
        if self.p > 0.:
            feat = self.drop(feat)

        x_pos = self.fc_pos(feat)
        x_ori = self.fc_ori(feat)
        
        loss = {}
        loss_tran = self.l2_loss(tran_labels, x_pos)
        loss_rot =  self.l2_loss(rot_labels, x_ori)
        loss_sum = self.hws_loss(loss_tran, loss_rot)

        loss['tran'] = loss_tran
        loss['rot'] = loss_rot
        loss['sum'] = loss_sum

        return loss, x_pos, x_ori
    
if __name__ == '__main__':
    from thop import profile
    model = DeepLO((3,64,2048))
    dummy_input = torch.zeros(4, 2, 8, 64, 2048)
    flops = model(dummy_input)