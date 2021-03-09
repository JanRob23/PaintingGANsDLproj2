import numpy as np
from numpy import asarray
from PIL import Image


def read_data(images_monet, images_photo):
    img_monet = []
    for image in images_monet:
        with open(image, 'rb') as file:
            img = asarray(Image.open(file))
            img_monet.append(img)
    img_photo = []
    for image in images_photo:
        with open(image, 'rb') as file:
            img = asarray(Image.open(file))
            img_photo.append(img)
    img_monet = np.array(img_monet)
    img_photo = np.array(img_photo)

    return img_monet, img_photo