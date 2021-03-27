import numpy as np
import imageio
from imgaug import augmenters as iaa
from numpy import asarray
from PIL import Image
from tqdm.notebook import tqdm
from tqdm import tqdm
import glob


def read_data(images_monet, images_photo):
    img_monet = []
    for image in tqdm(images_monet, desc='reading paintings'):
        with open(image, 'rb') as file:
            img = asarray(Image.open(file))
            img_monet.append(img)
    img_photo = []
    for image in tqdm(images_photo, desc = 'reading photos'):
        with open(image, 'rb') as file:
            img = asarray(Image.open(file))
            img_photo.append(img)
    img_monet = np.array(img_monet)
    img_photo = np.array(img_photo)

    return img_monet, img_photo

def flip(images):
    images = images.astype(np.uint8)
    newImages = list()
    for image in tqdm(images, desc='flipping'):
        flipH = np.fliplr(image).tolist()
        flipV = np.flipud(image).tolist()
        newImages.append(flipH)
        newImages.append(flipV)
    newImages = np.array(newImages)
    return np.concatenate((images, newImages), axis = 0)

def contrast(images):
    images = images.astype(np.uint8)
    newImages = list()
    contrast1=iaa.GammaContrast(gamma=0.5) # bright image
    contrast2=iaa.GammaContrast(gamma=2) # dark image
    for image in tqdm(images, desc='contrast'):
        bright = contrast1.augment_image(image.astype(np.uint8))
        dark = contrast2.augment_image(image.astype(np.uint8))
        newImages.append(bright)
        newImages.append(dark)
    newImages = np.array(newImages)
    return np.concatenate((images, newImages), axis = 0)

def crop(images):
    images = images.astype(np.uint8)
    newImages = list()
    crop = iaa.Crop(percent=(0, 0.2)) # crop image
    for image in tqdm(images, desc='cropping'):
        cropped=crop.augment_image(image)
        newImages.append(cropped)
    newImages = np.array(newImages)
    return np.concatenate((images, newImages), axis = 0)


def augment(images):
    images = images.astype(np.uint8)
    flipped = flip(images)
    cropped = crop(flipped)
    contrasted = contrast(cropped)
    for i, newImage in enumerate(contrasted):
        imageio.imwrite("Data/augs/%d.jpg" % (i,), newImage.astype(np.uint8))

def combinePhotos(paintings, photos):
    #5400 monet paintings, 7038 photos
    ypaint = [-1] * 5400
    yphoto = [1] * 7038
    y = np.array(ypaint + yphoto)

    #combine both image sets
    x = np.concatenate((paintings, photos), axis = 0)
    return x, y 

def getData():
    images_photo = glob.glob("Data/photo_jpg/*.jpg")
    augs = glob.glob("Data/augs/*.jpg")
    paintings, photos = read_data(augs, images_photo)
    print(f'paintings: {paintings.shape}\nphotos: {photos.shape}')
    return combinePhotos(paintings, photos)

