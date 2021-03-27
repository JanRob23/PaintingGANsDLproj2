import os
import time
import torch
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
            nn.MaxPool2d(6, 2), # 240 - 6 / 2 + 1 -> 119
            nn.Conv2d(64, 128, 6, 1, 0), # 119 - 6 + 1 -> 114
            nn.ReLU(),
            nn.MaxPool2d(4, 2), # 114 -4 /2 + 1 -> 56
            nn.Conv2d(128, 256, 4, 2), # 56 - 6 / 2 + 1 -> 26
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), #54
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2), # 110
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2, 0), #226
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 16, 1), #241
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 18, 1), #256
            nn.Sigmoid()
        )
        self.opt = torch.optim.Adam(self.parameters(),lr = start_lr, betas=(0.5, 0.999))
        self.epoch = 0
        self.training_range = range(self.epoch ,self.epochs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train(self, image_dl):
        for epoch in self.training_range:
            self.epoch = epoch
            start_time = time.time()
            avg_loss = 0.0
            if self.device == 'cuda':
                t = tqdmn(image_dl, leave=False, total=image_dl.__len__())
            else:
                t = tqdm(image_dl, leave=False, total=image_dl.__len__())
            
            for i, (photo_real, monet_real) in enumerate(t):
                photo_img, monet_img = photo_real.to(self.device), monet_real.to(self.device)
                update_req_grad(self, False)
                self.opt.zero_grad()
                fake_monet = forward(photo_img)
                loss = self.mse_loss(fake_monet, monet_img)
                loss.backward()
                self.opt.step()
                train_loss += loss.item()

            save_dict = {
            'epoch': epoch+1,
            'weights': self.state_dict(),
            'optimizer': self.opt.state_dict()
            }
            save_checkpoint(save_dict, 'current.ckpt')
            train_loss /= image_dl.__len__()
            time_req = time.time() - start_time
            self.loss_stats.append(train_loss, time_req)
            print(f'Epoch: {epoch+1} | Loss:{train_loss}')