import os
import time
import torch
from torch._C import TracingState
import torchvision
from torch import nn

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
            Conv2dSameBlock(64, 64),
            nn.MaxPool2d(6, 2), # 240 - 6 / 2 + 1 -> 119
            nn.Conv2d(64, 128, 6, 1, 0), # 119 - 6 + 1 -> 114
            nn.ReLU(),
            Conv2dSameBlock(128, 128),
            nn.MaxPool2d(4, 2), # 114 -4 /2 + 1 -> 56
            nn.Conv2d(128, 128, 4, 2), # 56 - 6 / 2 + 1 -> 26
            Conv2dSameBlock(128, 128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2), #54
            nn.ReLU(),
            Conv2dSameBlock(128, 64),
            nn.ConvTranspose2d(64, 64, 4, 2), # 110
            nn.ReLU(),
            Conv2dSameBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 6, 2, 0), #226
            nn.ReLU(),
            Conv2dSameBlock(32, 32),
            nn.ConvTranspose2d(32, 16, 16, 1), #241
            nn.ReLU(),
            Conv2dSameBlock(16, 16),
            nn.ConvTranspose2d(16, 3, 18, 1), #256
            Conv2dSameBlock(3, 3, relu=False),
            nn.Tanh()
        )
        self.opt = torch.optim.Adam(self.parameters(),lr = start_lr, betas=(0.5, 0.999))
        self.epoch = 0

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def load_model(self, ckpt):
        self.epoch = ckpt['epoch']
        self.load_state_dict(ckpt['weights'])
        self.opt.load_state_dict(ckpt['optimizer'])


    def train(self, image_dl):
        training_range = range(self.epoch, self.epochs) 
        for epoch in training_range:
            self.epoch = epoch
            start_time = time.time()
            avg_loss = 0.0
            if self.device == 'cuda':
                t = tqdmn(image_dl, leave=False, total=image_dl.__len__()- 1)
            else:
                t = tqdm(image_dl, leave=False, total=image_dl.__len__()- 1)
            
            for i, (photo_real, monet_real) in enumerate(t):
                photo_img, monet_img = photo_real.float().to(self.device), monet_real.float().to(self.device)
                #update_req_grad([self.encoder, self.decoder], False)
                self.opt.zero_grad()
                fake_monet = self.forward(photo_img)
                loss = self.mse_loss(fake_monet, monet_img)
                loss.backward()
                self.opt.step()
                avg_loss += loss.item()

            save_dict = {
            'epoch': epoch+1,
            'weights': self.state_dict(),
            'optimizer': self.opt.state_dict()
            }
            save_checkpoint(save_dict, 'current.ckpt')
            avg_loss /= image_dl.__len__()
            time_req = time.time() - start_time
            self.loss_stats.append(avg_loss, time_req)
            print(f'Epoch: {epoch+1} | Loss:{avg_loss}')

class Conv2dSameBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super(Conv2dSameBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = relu

    def forward(self, x):
        x = self.block(x)
        if self.relu:
            x = F.relu(x)
        return x