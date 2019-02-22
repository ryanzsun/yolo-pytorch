import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size = 1,
                               stride = 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size = 3,
                               stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(channels * 2)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = x + residual
        return x


class Darknet(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors, block, layers):
        super(Darknet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 3, 
                               stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1])
        self.layer3 = self._make_layer(block, 128, layers[2])
        self.layer4 = self._make_layer(block, 256, layers[3])
        self.layer5 = self._make_layer(block, 512, layers[4])

        self.out = nn.Conv2d(1024, num_anchors * (4 + 1 + num_classes), kernel_size = 1, stride = 1)

    def _make_layer(self, block, channels, blocks):
        downsample = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size = 3, 
                      stride = 2, padding = 1),
            nn.BatchNorm2d(channels * 2)
        )
        layers = [downsample]
        layers.append(block(channels))
        for _ in range(blocks-1):
            layers.append(block(channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.out(x)

        return x

        

net = Darknet(3, 20, 3, ResidualBlock, [1,2,8,8,4])
d = torch.rand((1,3,416,416))
print(net(d).shape)