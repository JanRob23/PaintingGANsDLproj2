from PIL import Image
from imgaug.imgaug import show_grid
import matplotlib.pyplot as plt
import glob
import numpy as np
from torch._C import DeviceObjType
from tqdm.notebook import tqdm
from tqdm import tqdm as cputqdm
import torch
from numpy import asarray
import os.path
from torch.utils.data import DataLoader
from utils import *
from fileIO import *
from AE import *
 

def go(monet, photos, train = True):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(719)
    img_ds = ImageDataset(monet, photos)
    img_dl = DataLoader(img_ds, batch_size=1, pin_memory=True)
    photo_img, monet_img = next(iter(img_dl))

    ae = autoencoder(30, device)
    ae.to(device)
    torch.set_grad_enabled(True)

    save_dict = {
        'epoch': 0,
        'weights': ae.state_dict(),
        'optimizer': ae.opt.state_dict()
    }
    save_checkpoint(save_dict, 'init.ckpt')

    # if os.path.isfile('current.ckpt'):
    #     if device == 'cpu':
    #         ae.load_model(load_checkpoint('current.ckpt', map_location = torch.device('cpu')))
    #     else:
    #         ae.load_model(load_checkpoint('current.ckpt'))
        
    # print(ae.epoch)
    # if train:
    #     ae.train(img_dl)

    # plt.xlabel("Epochs")
    # plt.ylabel("Losses")
    # plt.plot(ae.loss_stats.losses, 'b', label='Loss')
    # plt.legend()
    # plt.show()

    # _, ax = plt.subplots(5, 2, figsize=(12, 12))
    # for i in range(5):
    #     photo_img, _ = next(iter(img_dl))
    #     pred_monet = ae(photo_img.to(device)).cpu().detach()
    #     photo_img = unnorm(photo_img)
    #     pred_monet = unnorm(pred_monet)
        
    #     ax[i, 0].imshow(photo_img[0].permute(1, 2, 0))
    #     ax[i, 1].imshow(pred_monet[0].permute(1, 2, 0))
    #     ax[i, 0].set_title("Input Photo")
    #     ax[i, 1].set_title("Monet-esque Photo")
    #     ax[i, 0].axis("off")
    #     ax[i, 1].axis("off")
    # plt.show()

    # 1 ------------------------

    ae.load_model(load_checkpoint('Data/trainedModels/cycle20.ckpt', map_location = torch.device('cpu')))

    ph_ds = PhotoDataset('Data/test/')
    ph_dl = DataLoader(ph_ds, batch_size=1, pin_memory=True)
    trans = transforms.ToPILImage()

    if device == 'cpu':
        t = cputqdm(ph_dl, leave=False, total=ph_dl.__len__())
    else:
        t = tqdm(ph_dl, leave=False, total=ph_dl.__len__())
            
    for i, photo in enumerate(t):
        
        with torch.no_grad():
            pred_monet = ae(photo.to(device)).cpu().detach()
        pred_monet = unnorm(pred_monet)
        img = trans(pred_monet[0]).convert("RGB")
        img.save("Data/testResults/AEcycle/" + str(i+1) + ".jpg")

    ae.load_model(load_checkpoint('Data/trainedModels/wcg30.ckpt', map_location = torch.device('cpu')))

    ph_ds = PhotoDataset('Data/test/')
    ph_dl = DataLoader(ph_ds, batch_size=1, pin_memory=True)
    trans = transforms.ToPILImage()

    if device == 'cpu':
        t = cputqdm(ph_dl, leave=False, total=ph_dl.__len__())
    else:
        t = tqdm(ph_dl, leave=False, total=ph_dl.__len__())
            
    for i, photo in enumerate(t):
        
        with torch.no_grad():
            pred_monet = ae(photo.to(device)).cpu().detach()
        pred_monet = unnorm(pred_monet)
        img = trans(pred_monet[0]).convert("RGB")
        img.save("Data/testResults/AEwcg/" + str(i+1) + ".jpg")


    ae.load_model(load_checkpoint('Data/trainedModels/wcggp30.ckpt', map_location = torch.device('cpu')))


    ph_ds = PhotoDataset('Data/test/')
    ph_dl = DataLoader(ph_ds, batch_size=1, pin_memory=True)
    trans = transforms.ToPILImage()

    if device == 'cpu':
        t = cputqdm(ph_dl, leave=False, total=ph_dl.__len__())
    else:
        t = tqdm(ph_dl, leave=False, total=ph_dl.__len__())
            
    for i, photo in enumerate(t):
        
        with torch.no_grad():
            pred_monet = ae(photo.to(device)).cpu().detach()
        pred_monet = unnorm(pred_monet)
        img = trans(pred_monet[0]).convert("RGB")
        img.save("Data/testResults/AEwcggp/" + str(i+1) + ".jpg")

if __name__ == "__main__":
    monet = 'Data/customMonet'
    photos = 'Data/photo_jpg'
    go(monet, photos)