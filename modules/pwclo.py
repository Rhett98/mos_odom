#!/usr/bin/env python3
import os
import sys
path = os.getcwd()
sys.path.append(path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler

from modules.utils import quatt2T, transformPC

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True) 

class PyramidBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidBlock, self).__init__()

        self.conv1 = conv(in_channel, out_channel, kernel_size=3, stride=2)
        self.conv2 = conv(out_channel, out_channel, kernel_size=3, stride=1)
        self.conv3 = conv(out_channel, out_channel, kernel_size=3, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


class PWCNet(nn.Module):
    """ Odometry regression network"""
    def __init__(self):
        super(PWCNet, self).__init__()

        self.p_size = 9
        # feature pyramid networks
        self.down_conv1 = PyramidBlock(5, 16)
        self.down_conv2 = PyramidBlock(16, 32)
        self.down_conv3 = PyramidBlock(32, 64)
        self.down_conv4 = PyramidBlock(64, 96)
        self.down_conv5 = PyramidBlock(96, 128)
        self.down_conv6 = PyramidBlock(128, 196)
        
        # cost volume network
        self.correlation_sampler = SpatialCorrelationSampler(
                                                        kernel_size=1,
                                                        patch_size=self.p_size,
                                                        stride=1,
                                                        padding=0,
                                                        dilation_patch=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        # self.leakyRELU = nn.ReLU(True)
        
        # odom coarse
        self.pool = nn.AvgPool2d(2,2)
        self.q5_predict = nn.Conv1d(in_channels=64,out_channels=4,kernel_size=1)
        self.t5_predict = nn.Conv1d(in_channels=64,out_channels=3,kernel_size=1)
        self.q4_predict = nn.Conv1d(in_channels=256,out_channels=4,kernel_size=1)
        self.t4_predict = nn.Conv1d(in_channels=256,out_channels=3,kernel_size=1)
        self.q3_predict = nn.Conv1d(in_channels=1024,out_channels=4,kernel_size=1)
        self.t3_predict = nn.Conv1d(in_channels=1024,out_channels=3,kernel_size=1)
        self.q2_predict = nn.Conv1d(in_channels=4096,out_channels=4,kernel_size=1)
        self.t2_predict = nn.Conv1d(in_channels=4096,out_channels=3,kernel_size=1)
        # self.q3_predict = nn.Sequential(nn.Conv1d(in_channels=1024,out_channels=512,kernel_size=1),
        #                                 nn.Conv1d(in_channels=512,out_channels=4,kernel_size=1))
        # self.t3_predict = nn.Sequential(nn.Conv1d(in_channels=1024,out_channels=512,kernel_size=1),
        #                                 nn.Conv1d(in_channels=512,out_channels=3,kernel_size=1))
        # self.q2_predict = nn.Sequential(nn.Conv1d(in_channels=4096,out_channels=512,kernel_size=1),
        #                                 nn.Conv1d(in_channels=512,out_channels=4,kernel_size=1))
        # self.t2_predict = nn.Sequential(nn.Conv1d(in_channels=4096,out_channels=512,kernel_size=1),
        #                                 nn.Conv1d(in_channels=512,out_channels=3,kernel_size=1))
        self.q1_predict = nn.Sequential(nn.Conv1d(in_channels=16384,out_channels=4096,kernel_size=1),
                                        # nn.Conv1d(in_channels=4096,out_channels=512,kernel_size=1),
                                        nn.Conv1d(in_channels=4096,out_channels=4,kernel_size=1))
        self.t1_predict = nn.Sequential(nn.Conv1d(in_channels=16384,out_channels=4096,kernel_size=1),
                                        # nn.Conv1d(in_channels=4096,out_channels=512,kernel_size=1),
                                        nn.Conv1d(in_channels=4096,out_channels=3,kernel_size=1))
        
        # odom refine
        nd = self.p_size**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)
        self.downfeat6 = conv(8, 2, kernel_size=3, stride=2) 
        
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4]) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        self.downfeat5 = conv(8, 2, kernel_size=3, stride=2) 
        
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        self.downfeat4 = conv(8, 2, kernel_size=3, stride=2) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        self.downfeat3 = conv(8, 2, kernel_size=3, stride=2) 
        
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.downfeat2 = conv(8, 2, kernel_size=3, stride=2) 
        
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)
        self.downfeat1 = conv(8, 2, kernel_size=3, stride=2) 
        
        # init layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def corr(self, input1, input2):
        out_corr = self.correlation_sampler(input1.cpu(), input2.cpu())
        b, ph, pw, h, w = out_corr.size()
        out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
        return out_corr.cuda()
    
    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = torch.ones(x.size()).cuda()
        # if x.is_cuda:
        #     mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
        
        mask[mask<0.5] = 0
        mask[mask>0] = 1
        
        return output*mask
          
    # def predict_pose(self, x, y):
    #     """use svd to predict pose

    #     Args:
    #         x (tensor): (B, 3, H, W)
    #         y (tensor): (B, 3, H, W)
    #     Returns:
    #         Transformation: torch.Tensor (B, 4, 4) or (4, 4)
    #     """
    #     B, C, H, W = x.size()
    #     x_in = x.view(B, C, -1).permute(0, 2, 1)
    #     y_in = y.view(B, C, -1).permute(0, 2, 1)
    #     weight_svd = WeightedProcrustes()
    #     return weight_svd(x_in, y_in)
        
    def forward(self, x, y):
        x1 = F.interpolate(x[:,1:4], scale_factor=0.5, mode='bicubic')
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bicubic')
        x3 = F.interpolate(x2, scale_factor=0.5, mode='bicubic')
        x4 = F.interpolate(x3, scale_factor=0.5, mode='bicubic')
        x5 = F.interpolate(x4, scale_factor=0.5, mode='bicubic')

        y1 = F.interpolate(y[:,1:4], scale_factor=0.5, mode='bicubic')
        y2 = F.interpolate(y1, scale_factor=0.5, mode='bicubic')
        y3 = F.interpolate(y2, scale_factor=0.5, mode='bicubic')
        y4 = F.interpolate(y3, scale_factor=0.5, mode='bicubic')
        y5 = F.interpolate(y4, scale_factor=0.5, mode='bicubic')

        # feature pyramid
        f11 = self.down_conv1(x)
        f12 = self.down_conv2(f11)
        f13 = self.down_conv3(f12)
        f14 = self.down_conv4(f13)
        f15 = self.down_conv5(f14)
        f16 = self.down_conv6(f15)
        # print(f11.shape,f12.shape,f13.shape,f14.shape,f15.shape,f16.shape)
        f21 = self.down_conv1(y)
        f22 = self.down_conv2(f21)
        f23 = self.down_conv3(f22)
        f24 = self.down_conv4(f23)
        f25 = self.down_conv5(f24)
        f26 = self.down_conv6(f25)
        
        # pyramid 6 
        # cost volume
        corr6 = self.corr(f16, f26)
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x) # 2, 2, 64
        # odom predict
        in_feat6 = self.downfeat6(torch.cat((x5, y5, up_feat6), 1)).view(up_feat6.shape[0], -1, 1) # 1,128,1
        q5 = self.q5_predict(in_feat6).view(-1,4)
        t5 = self.t5_predict(in_feat6).view(-1,3)
        # with torch.no_grad():
        pose5 = quatt2T(t5,q5/torch.norm(q5))

        # pyramid 5 
        warp5 = self.warp(f15, up_flow6)
        corr5 = self.corr(warp5, f15) 
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, f25, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x) # 2,4,128
        # odom predict
        x4_warp = transformPC(x4, pose5)
        in_feat5 = self.downfeat5(torch.cat((x4_warp, y4, up_feat5),1)).view(up_feat5.shape[0], -1, 1) #512
        q4 = self.q4_predict(in_feat5).view(-1,4)
        t4 = self.t4_predict(in_feat5).view(-1,3)
        # with torch.no_grad():
        pose4 = torch.bmm(pose5, quatt2T(t4,q4/torch.norm(q4)))

        # pyramid 4 
        warp4 = self.warp(f14, up_flow5)
        corr4 = self.corr(warp4, f24)  
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, f24, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)
        # odom predict
        x3_warp = transformPC(x3, pose4)
        in_feat4 = self.downfeat4(torch.cat((x3_warp, y3, up_feat4),1)).view(up_feat4.shape[0], -1, 1) #
        # print("in_feat4",in_feat4)
        # print(in_feat4.shape)
        q3 = self.q3_predict(in_feat4).view(-1,4)
        t3 = self.t3_predict(in_feat4).view(-1,3)
        # with torch.no_grad():
        pose3 = torch.bmm(pose4, quatt2T(t3,q3/torch.norm(q3)))

        
        # pyramid 3 
        warp3 = self.warp(f13, up_flow4)
        corr3 = self.corr(warp3, f23) 
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, f23, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)
        # odom predict
        x2_warp = transformPC(x2, pose3)
        in_feat3 = self.downfeat3(torch.cat((x2_warp, y2, up_feat3),1)).view(up_feat3.shape[0], -1, 1)
        q2 = self.q2_predict(in_feat3).view(-1,4)
        t2 = self.t2_predict(in_feat3).view(-1,3)
        # with torch.no_grad():
        pose2 = torch.bmm(pose3, quatt2T(t2,q2/torch.norm(q2)))

        # pyramid 2 
        warp2 = self.warp(f12, up_flow3) 
        corr2 = self.corr(warp2, f22)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, f22, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
        
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        up_flow2 = self.deconv2(flow2)
        # odom predict
        x1_warp = transformPC(x1, pose4)
        in_feat2 = self.downfeat2(torch.cat((x1_warp, y1, up_flow2),1)).view(up_flow2.shape[0], -1, 1)
        # print("in_feat2",in_feat2)
        q1 = self.q1_predict(in_feat2).view(-1,4)
        t1 = self.t1_predict(in_feat2).view(-1,3)
        # with torch.no_grad():
        pose1 = torch.bmm(pose2, quatt2T(t1,q1/torch.norm(q1)))

        return pose1, pose2, pose3, pose4, pose5


# if __name__ == '__main__':
    # from thop import profile
    # model = PWCNet().cuda()
    # # dummy_input = torch.randn(1, 5, 64, 2048),torch.randn(1, 5, 64, 2048),torch.randn(1, 3),torch.randn(1, 4),
    # dummy_input = torch.randn(1, 5, 64, 2048).cuda(),torch.randn(1, 5, 64, 2048).cuda(),
    # flops, params = profile(model, (dummy_input))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    
