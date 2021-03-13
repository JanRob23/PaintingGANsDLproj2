from read_data import getData, read_data, augment
from jan_GAN.train import train, generate
import glob
# x, y = getData()
# print(x.shape, y.shape)
# paintings = glob.glob("Data/monet_jpg/*jpg")
# photos = glob.glob("Data/photo_jpg/*jpg")
# paintings, photos = read_data(paintings, photos)
#
# print(paintings.shape, photos.shape)
# photos = photos[0:7000]
# print(paintings.shape, photos.shape)
# model = train(paintings, photos)
# generate(photos, model)
def go(filepath_photos, filepath_paintings):
    photos = glob.glob(filepath_photos)
    paintings = glob.glob(filepath_paintings)
    print(filepath_paintings, filepath_photos)
    paintings, photos = read_data(paintings, photos)
    photos = photos[0:7000]
    print(paintings.shape, photos.shape)
    model = train(paintings, photos)
    return model
#go("Data/photo_jpg/*jpg", "Data/monet_jpg/*jpg")