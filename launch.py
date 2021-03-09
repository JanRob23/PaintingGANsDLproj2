import glob
from read_data import read_data

images_monet = glob.glob("Data/monet_jpg/*.jpg")
images_photo = glob.glob("Data/photo_jpg/*.jpg")
augs = glob.glob("Data/augs/*.jpg")
#paintings, photos = read_data(images_monet, images_photo)
paintings, photos = read_data(augs, images_photo)
