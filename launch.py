import glob
from read_data import read_data
images_monet = glob.glob("Data/monet_jpg/*.jpg")
images_photo = glob.glob("Data/photo_jpg/*.jpg")
monet_numpy, photo_numpy = read_data(images_monet, images_photo)
print(monet_numpy.shape, photo_numpy.shape)
