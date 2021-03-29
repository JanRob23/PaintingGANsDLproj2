import os
import time
import torch
import torchvision
from torch import nn
from collections import namedtuple
from torchvision import models
import itertools
from tqdm.notebook import tqdm as tqdmn
from tqdm import tqdm 
from utils import *
from fileIO import *

import torch.nn as nn
import torch.nn.functional as F 

class autoencoder(nn.Module):
    def __init__(self, epochs, device, start_lr=2e-4):
        super(autoencoder, self).__init__()
        self.epochs = epochs
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.loss_stats = AvgStats()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 17, 1, 0), # 256 - 17 + 1 -> 240
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), # 240 - 3 + 2 + 1 -> 240
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # 240 - 3 + 2 + 1 -> 240
            nn.ReLU(),
            nn.MaxPool2d(6, 2), # 240 - 6 / 2 + 1 -> 119
            nn.Conv2d(64, 128, 6, 1, 0), # 119 - 6 + 1 -> 114
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.MaxPool2d(4, 2), # 114 -4 /2 + 1 -> 56
            nn.Conv2d(128, 128, 4, 2), # 56 - 6 / 2 + 1 -> 26
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2), #54
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2), # 110
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2, 0), #226
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 16, 1), #241
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 18, 1), #256
            nn.Conv2d(3, 3, 3, 1, 1),  # keeps it same
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, 1, 1),  # keeps it same
            nn.Tanh()
        )
        self.opt = torch.optim.Adam(self.parameters(),lr = start_lr, betas=(0.5, 0.999))
        self.epoch = 0
        self.training_range = range(self.epoch ,self.epochs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out



class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False),
        )

    def forward(self, x):
        return self.block(x) + x


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2), nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x