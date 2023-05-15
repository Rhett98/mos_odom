#!/usr/bin/env python3
import os
import sys
path = os.getcwd()
sys.path.append(path)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FireConv(nn.Module):
    """ FireConv layer"""
    def __init__(self, inplanes: int, c1: int, c2: int, c3: int) -> None:
        super(FireConv, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(inplanes, c1, kernel_size=1)
        self.expand1x1 = nn.Conv2d(c1, c2, kernel_size=1)
        self.expand3x3 = nn.Conv2d(c1,c3, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.squeeze(x))
        return torch.cat([
            self.relu(self.expand1x1(x)),
            self.relu(self.expand3x3(x))], 1)
        
class SELayer(nn.Module):
    """ Squeeze and Excitation layer from SEnet (channel attention) """
    def __init__(self, in_features: int, reduction=16) -> None:
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 1x1 output size
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # BxC
        y = self.fc(y).view(b, c, 1, 1) # BxCx1x1
        x_scaled = x * y.expand_as(x) # BxCxHxW
        return x_scaled    

    
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=128):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), # conv1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate)) # conv w/ rate dilations

        modules.append(ASPPPooling(in_channels, out_channels)) # global average pooling

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), # conv1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.25))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

    
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

    
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
class MaskEncoder(nn.Module):
    """ mask prediction network encoder """
    def __init__(self, c: int) -> None:
        super(MaskEncoder, self).__init__()

        self.conv1 = nn.Conv2d(c, 64, kernel_size=3, stride=(1,2), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1,2), padding=1)
        self.fire1 = FireConv(64, 16, 64, 64)
        self.fire2 = FireConv(128, 16, 64, 64)
        self.se1 = SELayer(128, reduction=2)
        
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1,2), padding=1)
        self.fire3 = FireConv(128, 32, 128, 128)
        self.fire4 = FireConv(256, 32, 128, 128)
        self.se2 = SELayer(256, reduction=2)
        
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(1,2), padding=1)
        self.fire5 = FireConv(256, 48, 192, 192)
        self.fire6 = FireConv(384, 48, 192, 192)
        self.fire7 = FireConv(384, 64, 256, 256)
        self.fire8 = FireConv(512, 64, 256, 256)
        self.se3 = SELayer(512, reduction=2)
        
        # Enlargement layer
        self.aspp = ASPP(512, [6, 9, 12])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c1 = F.relu(self.conv1(x), inplace=True)
        x_p1 = self.pool1(x_c1)
        x_f1 = self.fire1(x_p1)
        x_f2 = self.fire2(x_f1)
        x_se1 = self.se1(x_f2)
        
        x_p2 = self.pool2(x_se1)
        x_f3 = self.fire3(x_p2)
        x_f4 = self.fire4(x_f3)
        x_se2 = self.se2(x_f4)
        
        x_p3 = self.pool3(x_se2)
        x_f5 = self.fire5(x_p3)
        x_f6 = self.fire6(x_f5)
        x_f7 = self.fire7(x_f6)
        x_f8 = self.fire8(x_f7)
        x_se3 = self.se3(x_f8)
        
        x_el = self.aspp(x_se3)
        return x_el
    
class OdomRegNet(nn.Module):
    """ Main odometry regression network - 2-stream net """
    def __init__(self, feature_channels=8):
        super(OdomRegNet, self).__init__()

        self.mask_encode = MaskEncoder(feature_channels) # [xyz range intensity normals]

        self.fire_1 = FireConv(256, 64, 256, 256)
        self.fire_2 = FireConv(512, 64, 256, 256)
        self.se_1 = SELayer(512, reduction=2)
        
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=(2,2), padding=1)
        self.fire_3 = FireConv(512, 80, 384, 384)
        self.fire_4 = FireConv(768, 80, 384, 384)
        
        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=(2,2), padding=1)
        self.fc1 = nn.Linear(393216, 512)
        self.dropout = nn.Dropout2d(p=0.5)
        
        self.fc2 = nn.Linear(512, 3)
        self.fc3 = nn.Linear(512, 4) 
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_mask_out = torch.cat([self.mask_encode(x), self.mask_encode(y)], 1) # B, C',H, W
        x_f1 = self.fire_1(x_mask_out) 
        x_f2 = self.fire_2(x_f1)
        x_se = self.se_1(x_f2)
        
        x_p1 = self.pool_1(x_se)
        x_f3 = self.fire_3(x_p1)
        x_f4 = self.fire_4(x_f3)
        
        x_p2 = self.pool_2(x_f4)
        x_p2 = x_p2.view(x_p2.size(0), -1) # flatten
        print(x_p2.shape)
        x_fc1 = self.dropout(self.fc1(x_p2))
        
        x_out = self.fc2(x_fc1) # translation x
        q_out = self.fc3(x_fc1) # rotation quarternion q
        return x_out, q_out
    
if __name__ == '__main__':
    from thop import profile
    model = OdomRegNet()
    dummy_input = torch.randn(1, 8, 64, 2048), torch.randn(1, 8, 64, 2048)
    flops, params = profile(model, (dummy_input))
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))