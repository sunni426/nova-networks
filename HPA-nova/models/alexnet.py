'''
    Self-defined model, based on Alexnet, for bottom-up training
    Mar 5

    References: 
        https://blog.paperspace.com/alexnet-pytorch/ 
'''

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from configs import Config

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import num_channels as config
class AlexNet(nn.Module):
    def __init__(self, num_classes=19, num_channels=4):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 96, kernel_size=5, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        # self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, cnt=1, gradcam=False):
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        cell = self.fc2(out)

        # # Referencing resnet.py code forward method
        # pooled = nn.Flatten()(self.pool(out))
        # # viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # # viewed_pooled = viewed_pooled.max(1)[0]

        # # Assume just returning one output here
        # return self.fc2(pooled)

        return cell

