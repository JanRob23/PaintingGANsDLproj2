import numpy as np


def flip(images):
    print(images[0].shape)
    print(images.shape)
    newImages = np.empty((1,256, 256, 3))
    for image in images:
        newImage = np.fliplr(image)
        newImages = np.append(newImages,newImage, axis = 1)
        newImage = np.flipud(image)
        newImages = np.append(newImages,newImage, axis = 1)
        print(newImages.shape)
    print('ddddddd',images.shape)
    np.concatenate((images, newImages), axis = 0)
    print(images.shape)

def augment(images):
    pass
