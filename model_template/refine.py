import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from model_template.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class Contextnet(nn.Module):
    def __init__(self, cc=16):
        super(Contextnet, self).__init__()
        c = cc
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]
    
class Unet(nn.Module):
    def __init__(self, cc=16):
        super(Unet, self).__init__()
        self.c = cc
        if self.c > 0:
            self.down0 = Conv2(17, 2*self.c)
            self.down1 = Conv2(4*self.c, 4*self.c)
            self.down2 = Conv2(8*self.c, 8*self.c)
            self.down3 = Conv2(16*self.c, 16*self.c)
            self.up0 = deconv(32*self.c, 8*self.c)
            self.up1 = deconv(16*self.c, 4*self.c)
            self.up2 = deconv(8*self.c, 2*self.c)
            self.up3 = deconv(4*self.c, self.c)
            self.conv = nn.Conv2d(self.c, 3, 3, 1, 1)
        else:
            self.conv = nn.Conv2d(17, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        if self.c > 0:
            s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
            s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
            s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
            s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
            x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
            x = self.up1(torch.cat((x, s2), 1)) 
            x = self.up2(torch.cat((x, s1), 1)) 
            x = self.up3(torch.cat((x, s0), 1)) 
            x = self.conv(x)
        else:
            x = self.conv(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        return torch.sigmoid(x)
