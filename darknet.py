import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.modules):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1,
                               stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,)
                               stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = x + residual
        return x


class Darknet(nn.modules):
    def __init__(self, in_channels, num_classes):
        super(Darknet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 3, 
                               stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(residual_block, , layers[0])
        self.layer2 = self._make_layer(residual_block, , layers[1])
        self.layer3 = self._make_layer(residual_block, , layers[2])
        self.layer4 = self._make_layer(residual_block, , layers[3])
        self.layer5 = self._make_layer(residual_block, , layers[4])

        