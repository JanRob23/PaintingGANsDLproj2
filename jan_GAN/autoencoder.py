

import torch
from torch import nn
from torch.autograd import Variable
from tqdm.notebook import tqdm
#from tqdm import tqdm
import os


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 16, 3, 0), # 81
            nn.Tanh(),
            nn.MaxPool2d(3, 2), # 40
            nn.Conv2d(32, 64, 6, 2, 0), # 18
            nn.Tanh(),
            nn.MaxPool2d(3, 1) # 16
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1), #31
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), # 62
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, 4, 2, 0), #126
            nn.Tanh(),
            nn.ConvTranspose2d(8, 3, 6, 2, 0), # 256
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(monet_images):
    num_epochs = 300
    batch_size = 10
    learning_rate = 2e-5
    model = autoencoder()
    monet_images = torch.from_numpy(monet_images.copy())
    monet_images = monet_images.reshape(-1, batch_size, 3, 256, 256)
    monet_images = monet_images.float()
    if torch.cuda.is_available():
        model = model.cuda()
        monet_images = monet_images.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    total_loss = 0
    for epoch in tqdm(range(num_epochs), desc='epochs'):
        for batch in range(monet_images.shape[0]):
            img = monet_images[batch]
            img = img / 255
            if torch.cuda.is_available():
                img = Variable(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        total_loss += loss.data
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data))
        # if epoch % 10 == 0:
        #     pic = to_img(output.cpu().data)
        #     save_image(pic, './dc_img/image_{}.png'.format(epoch))

    return model