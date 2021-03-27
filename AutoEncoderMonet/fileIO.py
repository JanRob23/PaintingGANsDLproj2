import numpy as np
import os
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, monet_dir, photo_dir, size=(256, 256), normalize=True):
        super().__init__()
        self.monet_dir = monet_dir
        self.photo_dir = photo_dir
        self.monet_idx = dict()
        self.photo_idx = dict()
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()                               
            ])
        for i, fl in enumerate(os.listdir(self.monet_dir)):
            self.monet_idx[i] = fl
        for i, fl in enumerate(sorted(os.listdir(self.photo_dir))):
            self.photo_idx[i] = fl
        self.idx = 1
        self.range = len(self.monet_idx.keys()) 

    def __getitem__(self, idx):
        photo_img = Image.open(f'{self.photo_dir}/photo{self.idx} .jpg')
        photo_img = self.transform(photo_img)
        monet_img = Image.open(f'{self.monet_dir}/{self.idx}.jpg')
        monet_img = self.transform(monet_img)
        if self.idx == self.__len__:
            self.idx =1
        else :
            self.idx +=1
        return photo_img, monet_img

    def __len__(self):
        return min(len(self.monet_idx.keys()), len(self.photo_idx.keys()))

def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt

def save_checkpoint(state, save_path):
    torch.save(state, save_path)

class PhotoDataset(Dataset):
    def __init__(self, photo_dir, size=(256, 256), normalize=True):
        super().__init__()
        self.photo_dir = photo_dir
        self.photo_idx = dict()
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()                               
            ])
        for i, fl in enumerate(sorted(os.listdir(self.photo_dir))):
            self.photo_idx[i] = fl
        self.idx = 1
    def __getitem__(self, idx):
        photo_path = os.path.join(self.photo_dir, self.photo_idx[idx])
        #photo_img = Image.open(photo_path)
        photo_img = Image.open(f'Data/photo_jpg/photo{self.idx} .jpg')
        photo_img = self.transform(photo_img)
        self.idx +=1
        return photo_img

    def __len__(self):
        return len(self.photo_idx.keys())