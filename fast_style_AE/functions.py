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
from models_AE_VGG import *
import torch.nn as nn
import torch.nn.functional as F
def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram

def train(image_dl, device):
    # HYPERPARAMETERS
    learning_rate = 2e-4
    lambda_content = 1e5
    lambda_style = 1e10
    ae = autoencoder(20, device)
    ae.to(device)
    vgg = VGG16()
    vgg.to(device)
    # Define optimizer and loss
    opt = torch.optim.Adam(ae.parameters(),lr = learning_rate, betas=(0.5, 0.999))
    l2_loss = torch.nn.MSELoss().to(ae.device)

    for epoch in ae.training_range:
        ae.epoch = epoch
        start_time = time.time()
        avg_loss = 0.0
        if ae.device == 'cuda':
            t = tqdmn(image_dl, leave=False, total=image_dl.__len__( )- 1)
        else:
            t = tqdm(image_dl, leave=False, total=image_dl.__len__( )- 1)

        for i, (content_style, monet_style) in enumerate(t):
            photo_img, monet_img = content_style.float().to(ae.device), monet_style.float().to(ae.device)
            # update_req_grad([self.encoder, self.decoder], False)
            opt.zero_grad()
            fake_monet = ae.forward(photo_img)
            if i < 2:
                used_monet = monet_img
            # get content loss
            features_original = vgg.forward(photo_img)
            features_transformed = vgg.forward(fake_monet)
            content_loss = lambda_content * l2_loss(features_original.relu2_2, features_transformed.relu2_2)
            # Extract style features
            features_style_original = vgg.forward(used_monet)
            style_loss = 0
            for ft_y, ft_s in zip(features_transformed, features_style_original):
                gm_y = gram_matrix(ft_y)
                gm_s = gram_matrix(ft_s)
                style_loss += l2_loss(gm_y, gm_s)
            style_loss = lambda_style * style_loss
            loss = style_loss + content_loss

            loss.backward()
            opt.step()
            avg_loss += loss.item()

        save_dict = {
            'epoch': epoch + 1,
            'weights': ae.state_dict(),
            'optimizer': ae.opt.state_dict()
        }
        save_checkpoint(save_dict, 'current.ckpt')
        avg_loss /= image_dl.__len__()
        time_req = time.time() - start_time
        ae.loss_stats.append(avg_loss, time_req)
        print(f'Epoch: {epoch +1} | Loss:{avg_loss}')

    return ae