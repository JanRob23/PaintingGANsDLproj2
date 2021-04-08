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
from functions import *
from models_AE_VGG import *
 

def go(monet, photos):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    set_seed(719)
    img_ds = ImageDataset(monet, photos)
    img_dl = DataLoader(img_ds, batch_size=1, pin_memory=True)
    photo_img, monet_img = next(iter(img_dl))
    ae = train(img_dl, device)
    torch.set_grad_enabled(True)
    save_dict = {
        'epoch': 0,
        'weights': ae.state_dict(),
        'optimizer': ae.opt.state_dict()
    }
    save_checkpoint(save_dict, 'init.ckpt')

    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.plot(ae.loss_stats.losses, 'b', label='Loss')
    plt.legend()
    plt.show()

    _, ax = plt.subplots(5, 2, figsize=(12, 12))
    for i in range(5):
        photo_img, _ = next(iter(img_dl))
        inp = photo_img.to(device)
        pred_monet = ae(inp).cpu().detach()
        photo_img = unnorm(photo_img)
        pred_monet = unnorm(pred_monet)
        ax[i, 0].imshow(photo_img[0].permute(1, 2, 0))
        ax[i, 1].imshow(pred_monet[0].permute(1, 2, 0))
        ax[i, 0].set_title("Input Photo")
        ax[i, 1].set_title("Monet-esque Photo")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
    plt.show()

    ''' Code to generate images with a trained model'''
    # ph_ds = PhotoDataset('Data/testtest/')
    # ph_dl = DataLoader(ph_ds, batch_size=1, pin_memory=True)
    # trans = transforms.ToPILImage()
    # ae = TransformerNet(0.0001)
    # #ae = autoencoder(10, device)
    # ae.load_model(load_checkpoint('Data/high_content_loss.ckpt', map_location=torch.device('cpu')))
    # t = cputqdm(ph_dl, leave=False, total=ph_dl.__len__())
    # for i, photo in enumerate(t):
    #     with torch.no_grad():
    #         pred_monet = ae.forward(photo.to(device)).cpu().detach()
    #     pred_monet = denormalize(pred_monet)
    #     plt.imshow(pred_monet[0].permute(1, 2, 0))
    #     plt.imshow(photo[0].permute(1, 2, 0))
    #     #plt.show()
    #     img = pred_monet[0].permute(1, 2, 0)
    #     img = np.array(img)
    #     img = np.clip(img, 0, 1)
    #     plt.imsave("Data/testResults/" + str(i + 1) + ".jpg", img)

if __name__ == "__main__":
    photos = 'Data/photo_jpg'
    monet = 'Data/monet_jpg'
    go(monet, photos)