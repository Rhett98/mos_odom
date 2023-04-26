# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.BaseBlocks import ResBlock, UpBlock,ResContextBlock

class SalsaNext(nn.Module):
    def __init__(self, nclasses=3, input_scan=2):
        super(SalsaNext, self).__init__()
        self.nclasses = nclasses
        self.input_size = 5 * input_scan

        print("Depth of backbone input = ", self.input_size)
        ###
        
        self.downCntx = ResContextBlock(self.input_size, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)
        #  
        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c,down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)

        logits = logits
        logits = F.softmax(logits, dim=1)
        return logits
    
class SalsaNextEncoder(nn.Module):
    def __init__(self, nclasses=20, input_scan=1):
        super(SalsaNextEncoder, self).__init__()

        self.nclasses = nclasses
        self.input_size = 5 * input_scan
        
        self.downCntx = ResContextBlock(self.input_size, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

    def forward(self, x):

        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, _ = self.resBlock3(down1c)
        down3c, _ = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)
        return down0b,down1b,down1c,down2c,down5c
    
if __name__ == '__main__':
    from thop import profile
    model = SalsaNext(20, 1)
    dummy_input = torch.randn(1, 5, 64, 2048)
    flops, params = profile(model, (dummy_input,))
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    out = model(dummy_input)
    for i in range(len(out)):
        print(out[i].shape)