from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch as th
import torch

class Unet_module(nn.Module):
    def __init__(self, kernel_size, channel_list, down_up='down'):
        super(Unet_module, self).__init__()
        self.conv1 = nn.Conv3d(channel_list[0], channel_list[1], kernel_size, 1, (kernel_size - 1) // 2)
        self.conv2 = nn.Conv3d(channel_list[1], channel_list[2], kernel_size, 1, (kernel_size - 1) // 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(channel_list[1])
        self.bn2 = nn.BatchNorm3d(channel_list[2])

        if down_up == 'down':
            self.sample = nn.MaxPool3d(2, 2)
        else:
            self.sample = nn.Sequential(nn.ConvTranspose3d(channel_list[2], channel_list[2],
                                                           kernel_size, 2, (kernel_size - 1) // 2, 1), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        next_layer = self.sample(x)

        return next_layer, x
    
class UnetImageCAS(nn.Module):
    def __init__(self, kernel_size,in_channel=2, dist=False):
        super(UnetImageCAS, self).__init__()

        self.dist = dist

        self.u1 = Unet_module(kernel_size, (in_channel, 32, 64))
        self.u2 = Unet_module(kernel_size, (64, 64, 128))
        self.u3 = Unet_module(kernel_size, (128, 128, 256), down_up='up')
        self.u4 = Unet_module(kernel_size, (384, 128, 128), down_up='up')
        self.u5 = Unet_module(kernel_size, (192, 64, 64), down_up='up')

        self.u6=nn.Sequential(nn.Conv3d(64,32,3,1,1,bias=False),nn.ReLU(inplace=True),nn.BatchNorm3d(32))
        self.last_conv = nn.Conv3d(32, 1, 1, 1,bias=False)
        # self.activate_fun=nn.Sigmoid()
        self.dropout1=nn.Dropout3d(0.5)
        self.dropout2=nn.Dropout3d(0.5)

        ########################################################
        # if dist=True execute this line
        self.dist_conv = nn.Conv3d(32, 1, 1, 1,bias=False)
        ########################################################

    def forward(self, x, train=False):
        x, x_c1 = self.u1(x)
        x, x_c2 = self.u2(x)
        x, x1 = self.u3(x)
        x = th.cat([x, x_c2], dim=1)
        x, _ = self.u4(x)
        x = th.cat([x, x_c1], dim=1)
        _, x = self.u5(x)
        x=self.u6(x)
        # x=self.dropout1(x)
        x_seg = self.last_conv(x)
        #######################################################
        # if self.dist=True, output additional data (dist)
        x_dist = self.dist_conv(x)

        return x_seg, x_dist
  


