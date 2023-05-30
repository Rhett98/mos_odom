import os
import sys
path = os.getcwd()
sys.path.append(path)
import torch
from torch import nn
from torch.nn import functional as F

from modules.nets.pointseg_modules import Fire, SELayer
from modules.nets.pointseg_net import PSEncoder
from modules.nets.resnet import ResNetEncoder

def num_flat_features(x, dim=1):
    size = x.size()[dim:]  # all dimensions except the dim (e.g. dim=1 batch, dim=2 seq. )
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def eval_output_size_detection(model, input_shape):
    # in-feature size autodetection
    model.eval()
    c, h, w = input_shape
    with torch.no_grad():
        x = torch.randn((1, c, h, w))
        x = model(x)
        _, c, h, w = x.shape
    return c, h, w



def conv( batch_norm, in_planes, out_planes, kernel_size=(3, 3), stride=1):
    padding_h = (kernel_size[0] - 1) // 2
    padding_w = (kernel_size[1] - 1) // 2

    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(padding_h, padding_w),
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(padding_h, padding_w),
                      bias=True),
            nn.LeakyReLU()
        )

class LidarPointSegFeat(nn.Module):
    def __init__(self, input_shape, bn_d=0.1):
        super(LidarPointSegFeat, self).__init__()
        self.bn_d = bn_d
        self.p = 0.1
        c, h, w = input_shape
        self.fusion = 'cat'
        self.encoder1 = PSEncoder((2 * c, h, w))
        self.encoder2 = PSEncoder((2 * c, h, w))

        # shapes of  x_1a, x_1b, x_se1, x_se2, x_se3, x_el
        enc_out_shapes = self.encoder1.get_output_shape()

        # number of output channels in encoder
        b, c, h, w = enc_out_shapes

        #self.fire12 = nn.Sequential(Fire(768, 96, 384, 384, bn=True, bn_d=self.bn_d, bypass=False),
        #                            Fire(768, 96, 384, 384, bn=True, bn_d=self.bn_d, bypass=False))

        self.fc1 = nn.Linear(1536, 256)

        if self.p > 0.:
            self.drop = nn.Dropout(self.p)

        # self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """
        :param inputs: images of dimension [BxTxCxHxW], where T is seq-size+1, e.g. 2+1
        :return: outputs: features of dim [BxTxN]
        mask0: predicted mask to each time sequence
        """
        imgs_xyz, imgs_normals = x[:,:,1:4,:,:], x[:,:,5:,:,:]
        b, s, c, h, w = imgs_xyz.shape
        imgs_xyz = imgs_xyz.reshape(b, s*c, h, w)
        imgs_normals = imgs_normals.reshape(b, s*c, h, w)

        x_feat_0 = F.adaptive_avg_pool2d(self.encoder1(imgs_xyz), (1, 1)).flatten(1)
        x_feat_1 = F.adaptive_avg_pool2d(self.encoder2(imgs_normals), (1, 1)).flatten(1)

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        elif self.fusion == 'add':
            x = x_feat_0 + x_feat_1
        else:
            x = x_feat_0 - x_feat_1

        x = F.leaky_relu(self.fc1(x), inplace=False)
        # x = self.fc1(x)

        if self.p > 0.:
            x = self.drop(x)

        # reshape output to BxTxCxHxW
        x = x.view(b, num_flat_features(x, 1))
        return x


class LidarFlowNetFeat(nn.Module):
    def __init__(self, input_shape):
        super(LidarFlowNetFeat, self).__init__()
        c, h, w = input_shape
        batch_norm = True
        self.p = 0.
        self.fusion = 'cat'
        self.encoder1 = FlowNetEncoder([2*c, h, w])
        self.encoder2 = FlowNetEncoder([2*c, h, w])

        if self.p > 0:
            self.drop = nn.Dropout(self.p)

        self.fc1 = nn.Linear(2048, 256)
        # self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """
        :param inputs: images of dimension [BxTxCxHxW], where T is seq-size+1, e.g. 2+1
        :return: outputs: features of dim [BxTxN]
        mask0: predicted mask to each time sequence
        """
        imgs_xyz, imgs_normals = x[:,:,1:4,:,:], x[:,:,5:,:,:]

        b, s, c, h, w = imgs_xyz.shape
        imgs_xyz = imgs_xyz.reshape(b, s*c, h, w)
        imgs_normals = imgs_normals.reshape(b, s*c, h, w)

        x_feat_0 = self.encoder1(imgs_xyz)
        # print("********************************************")
        x_feat_1 = self.encoder2(imgs_normals)

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        elif self.fusion == 'add':
            x = x_feat_0 + x_feat_1
        else:
            x = x_feat_0 - x_feat_1

        x = F.leaky_relu(self.fc1(x), inplace=False)

        if self.p > 0.:
            x = self.drop(x)

        # reshape output to BxTxCxHxW
        x = x.view(b, num_flat_features(x, 1))
        return x


class LidarResNetFeat(nn.Module):
    def __init__(self, input_shape):
        super(LidarResNetFeat, self).__init__()
        self.p = 0
        self.fusion = 'cat'
        c, h, w = input_shape
        self.encoder1 = ResNetEncoder([2*c, h, w])
        self.encoder2 = ResNetEncoder([2*c, h, w])

        if self.p > 0:
            self.drop = nn.Dropout(self.p)

        self.fc1 = nn.Linear(1024, 256)

        # self.output_shape = self.calc_output_shape()

    def forward(self, x):
        imgs_xyz, imgs_normals = x[:,:,1:4,:,:], x[:,:,5:,:,:]

        b, s, c, h, w = imgs_xyz.shape
        imgs_xyz = imgs_xyz.reshape(b, s*c, h, w)
        imgs_normals = imgs_normals.reshape(b, s*c, h, w)

        x_feat_0 = self.encoder1(imgs_xyz)
        x_feat_1 = self.encoder2(imgs_normals)

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        elif self.fusion == 'add':
            x = x_feat_0 + x_feat_1
        else:
            x = x_feat_0 - x_feat_1

        if self.p > 0.:
            x = self.drop(x)

        x = F.leaky_relu(self.fc1(x))

        # reshape output to BxTxCxHxW
        x = x.view(b, num_flat_features(x, 1))
        return x


class LidarSimpleFeat1(nn.Module):
    def __init__(self, input_shape):
        super(LidarSimpleFeat1, self).__init__()
        bypass = False
        c, h, w = input_shape
        self.p = 0.25
        self.fusion = 'cat'
        self.encoder1 = FeatureNetSimple1([2*c, h, w], bypass=bypass)
        self.encoder2 = FeatureNetSimple1([2*c, h, w], bypass=bypass)

        if self.p > 0:
            self.drop = nn.Dropout(self.p)

        self.fc1 = nn.Linear(1024, 128)

        # self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """
        :param inputs: images of dimension [BxSxCxHxW], S:=Seq-length 
        :return: outputs: features of dim [BxN]
        mask0: predicted mask to each time sequence
        """
        imgs_xyz, imgs_normals = x[:,:,1:4,:,:], x[:,:,5:,:,:]

        b, s, c, h, w = imgs_xyz.shape
        imgs_xyz = imgs_xyz.reshape(b, s*c, h, w)
        imgs_normals = imgs_normals.reshape(b, s*c, h, w)

        x_feat_0 = self.encoder1(imgs_xyz)
        x_feat_1 = self.encoder2(imgs_normals)

        if self.fusion == 'cat':
            y = torch.cat((x_feat_0, x_feat_1), dim=1)
        elif self.fusion == 'add':
            y = x_feat_0 + x_feat_1
        else:
            y = x_feat_0 - x_feat_1

        if self.p > 0.:
            y = self.drop(y)

        y = F.leaky_relu(self.fc1(y[:, :, 0, 0]))
        # reshape output to BxTxCxHxW
   
        y = y.view(b, num_flat_features(y, 1))
        return y


class FlowNetEncoder(nn.Module):
    """Simple Conv. based Feature Network
    """
    def __init__(self, input_shape, batch_norm=True):
        super(FlowNetEncoder, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape

        self.conv1 = conv(batch_norm, c, 64, kernel_size=(5, 7), stride=(1, 2))
        self.conv2 = conv(batch_norm, 64, 128, kernel_size=(3, 5), stride=(1, 2))
        self.conv3 = conv(batch_norm, 128, 256, kernel_size=(3, 5), stride=(1, 2))
        self.conv3_1 = conv(batch_norm, 256, 256)
        self.conv4 = conv(batch_norm, 256, 512, stride=2)
        self.conv4_1 = conv(batch_norm, 512, 512)
        self.conv5 = conv(batch_norm, 512, 512, stride=2)
        self.conv5_1 = conv(batch_norm, 512, 512)
        self.conv6 = conv(batch_norm, 512, 1024, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        # print("out_conv2",out_conv2)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        # print("out_conv4",out_conv4)
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        out = self.pool(out_conv6)
        # print("out after pool",out)
        out = torch.flatten(out, 1)
        return out


class FeatureNetSimple1(nn.Module):
    """Simple Conv. based Feature Network with optinal bypass connections"""
    def __init__(self, input_shape, bypass=False):
        super(FeatureNetSimple1, self).__init__()

        self.bypass = bypass
        self.input_shape = input_shape
        c, h, w = self.input_shape

        self.conv1 = nn.Conv2d(c, out_channels=64, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv2 = nn.Conv2d(64, out_channels=128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv3 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, out_channels=256, kernel_size=3, stride=(1, 1), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)

        self.conv5 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=(1, 1), padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, out_channels=512, kernel_size=3, stride=(1, 1), padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)

        self.conv7 = nn.Conv2d(512, out_channels=512, kernel_size=3, stride=(1, 1), padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        # self.pool7 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)
        self.pool7 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 1. block
        out = F.relu(self.conv1(x), inplace=True)
        out = self.bn1(out)
        out = self.pool1(out)

        # 2. block
        out = F.relu(self.conv2(out), inplace=True)
        out = self.bn2(out)
        out = self.pool2(out)

        # 3. block
        out = F.relu(self.conv3(out), inplace=True)
        out = self.bn3(out)
        identitiy = out

        out = F.relu(self.conv4(out), inplace=True)
        out = self.bn4(out)
        if self.bypass:
            out += identitiy
        out = self.pool4(out)

        # 4. block
        out = F.relu(self.conv5(out), inplace=True)
        out = self.bn5(out)
        identitiy = out

        out = F.relu(self.conv6(out), inplace=True)
        out = self.bn6(out)
        if self.bypass:
            out += identitiy
        out = self.pool6(out)

        out = F.relu(self.conv7(out), inplace=True)
        out = self.bn7(out)
        out = self.pool7(out)
        return out

if __name__ == '__main__':
    from thop import profile
    # model = LidarPointSegFeat((3,64,2048))
    # model = LidarFlowNetFeat((3,64,2048))
    model = LidarResNetFeat((3,64,2048))
    # model = LidarSimpleFeat1((3,64,2048))
    dummy_input = torch.zeros(4, 2, 8, 64, 2048)
    flops = model(dummy_input)
    # flops, params = profile(model=model, inputs=(dummy_input))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))