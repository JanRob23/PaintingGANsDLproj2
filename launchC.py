from read_data import getData
from Model.Charlie.GAN import Gan

# x, y = getData()
# print(x.shape, y.shape)
def go():
    gan = Gan()
    gan.train()