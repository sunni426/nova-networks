'''
    Self-defined model, based on Efficientnet, for bottom-up training
    Mar 5

    References: 
        https://medium.com/@aniketthomas27/efficientnet-implementation-from-scratch-in-pytorch-a-step-by-step-guide-a7bb96f2bdaa
'''

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from configs import Config
from math import ceil

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

basic_mb_params = [
    # k, channels(c), repeats(t), stride(s), kernel_size(k)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

alpha, beta = 1.2, 1.1

scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, groups=groups),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU())

    def forward(self, x):
        return self.cnnblock(x)
        
class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride, padding, expand_ratio, reduction=2,
    ):
        super(MBBlock, self).__init__()
        # self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim

        # This is for squeeze and excitation block
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim,
                kernel_size=3,stride=1,padding=1)

        self.conv = nn.Sequential(
                ConvBlock(hidden_dim,hidden_dim,kernel_size,
                  stride,padding,groups=hidden_dim),
                SqueezeExcitation(hidden_dim, reduced_dim),
                nn.Conv2d(hidden_dim, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs):
        if self.expand:
          x = self.expand_conv(inputs)
        else:
          x = inputs
        return self.conv(x)
        
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class EfficientNet2(nn.Module):
    def __init__(self, model_name, num_classes=19, num_channels=4):
        super(EfficientNet2, self).__init__()
        phi, resolution, dropout = scale_values[model_name]
        self.num_channels = num_channels
        self.depth_factor, self.width_factor = alpha**phi, beta**phi
        self.last_channels = ceil(1280 * self.width_factor)
        self.avgpool= nn.AdaptiveAvgPool2d(1)
        self.feature_extractor()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channels, num_classes),
        )

    def feature_extractor(self):
        channels = int(32 * self.width_factor)
        features = [ConvBlock(self.num_channels, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for k, c_o, repeat, s, n in basic_mb_params:
            # For numeric stability, we multiply and divide by 4
            out_channels = 4 * ceil(int(c_o * self.width_factor) / 4)
            num_layers = ceil(repeat * self.depth_factor)

            for layer in range(num_layers):
                if layer == 0:
                  stride = s
                else:
                  stride = 1
                features.append(
                        MBBlock(in_channels,out_channels,expand_ratio=k,
                        stride=stride,kernel_size=n,padding=n// 2)
                    )
                in_channels = out_channels

        features.append(
            ConvBlock(in_channels, self.last_channels, 
            kernel_size=1, stride=1, padding=0)
        )
        self.extractor = nn.Sequential(*features)

    def forward(self, x, num_cells=10, gradcam=True):
        x = self.avgpool(self.extractor(x))
        cells = self.classifier(self.flatten(x))

        if not gradcam:
            x_pooled = self.flatten(x).view(-1, num_cells, self.flatten(x).shape[-1])
            x_pooled = x_pooled.max(1)[0]
            img = self.classifier(self.flatten(x_pooled))
            # print(f'cells: {cells.shape}, img: {img.shape}') # predicted_img: (90, 1, 19),predicted_cell (90, 10, 19), Mar 8
            return cells, img
        else:
            return cells, cells

# model_name = 'b1'
# output_class = 1000 #for imagenet
# effnet = EfficientNet(model_name, output_class)

