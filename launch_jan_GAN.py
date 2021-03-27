from read_data import getData, read_data, augment
from jan_GAN.train import train, generate
from jan_GAN.autoencoder import train_autoencoder
import glob

def go(filepath_photos, filepath_paintings):
    photos = glob.glob(filepath_photos)
    paintings = glob.glob(filepath_paintings)
    print(filepath_paintings, filepath_photos)
    paintings, photos = read_data(paintings, photos)
    photos = photos[0:500]
    print(paintings.shape, photos.shape)
    model = train(paintings, photos)
    return model
#go("Data/photo_jpg/*jpg", "Data/monet_jpg/*jpg")

def go_autoencoder(filepath_monet):
    paintings = glob.glob(filepath_monet)
    paintings, _ = read_data(paintings, [])
    paintings = paintings[0:7000]
    model = train_autoencoder(paintings)
    return model

#go_autoencoder("Data/monet_jpg/*jpg")