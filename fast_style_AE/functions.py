import os
import time
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import itertools
from tqdm.notebook import tqdm as tqdmn
from tqdm import tqdm
from PIL import Image
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
    learning_rate = 2e-5
    lambda_content = 1e5
    lambda_style = 1e5
    ae = autoencoder(15, device)
    ae.to(device)
    transformer_net = TransformerNet()
    transformer_net.to(device)
    vgg = VGG16()
    vgg.to(device)
    # Define optimizer and loss
    opt = torch.optim.Adam(transformer_net.parameters(),lr = learning_rate, betas=(0.5, 0.999))
    l2_loss = torch.nn.MSELoss().to(ae.device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
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
            fake_monet = transformer_net.forward(photo_img)
            if i == 0 and epoch == 0:
                used_monet = monet_img
                features_style = vgg.forward(used_monet)
                gram_style = [gram_matrix(y) for y in features_style]
                print("Monet used for transfer")
                monet_img = monet_img.cpu().detach()
                monet_img = unnorm(monet_img)
                plt.imshow(monet_img[0].permute(1, 2, 0))
                plt.show()
            # get content loss
            features_original = vgg.forward(photo_img)
            features_transformed = vgg.forward(fake_monet)
            content_loss = lambda_content * l2_loss(features_original.relu2_2, features_transformed.relu2_2)
            # Extract style features
            style_loss = 0
            for ft_y, gm_s in zip(features_transformed, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += l2_loss(gm_y, gm_s[: photo_img.size(0), :, :])
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
        transformer_net.loss_stats.append(avg_loss, time_req)
        print(f'Epoch: {epoch +1} | Loss:{avg_loss}')

    return transformer_net