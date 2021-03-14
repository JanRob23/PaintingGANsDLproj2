import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import imageio
from PIL import Image
from jan_GAN.model import Generator, Discriminator, initialize_weights

def train(paintings, photos):
    # Hyperparameters etc.
    LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
    NUM_EPOCHS = 10
    batch_size = 1

    gen = Generator(3, 64)
    disc = Discriminator(3, 64)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    paintings = torch.from_numpy(paintings.copy())
    photos = torch.from_numpy(photos.copy())
    paintings = paintings.reshape(-1, batch_size, 3, 256, 256)
    photos = photos.reshape(-1, batch_size, 3, 256, 256)
    paintings = paintings.float()
    photos = photos.float()
    if torch.cuda.is_available():
        gen = gen.cuda()
        disc = disc.cuda()
        photos = photos.cuda()
        paintings = paintings.cuda()
    print(paintings.shape[0])
    for epoch in tqdm(range(NUM_EPOCHS), desc='Epochs'):
        for batch in tqdm(range(paintings.shape[0]), desc='Bitches'):
            painting = paintings[batch]
            photo = photos[batch]
            fake_painting = gen(photo)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_painting = disc(painting).reshape(-1)
            loss_disc_painting = criterion(disc_painting, torch.ones_like(disc_painting))
            disc_fake = disc(fake_painting.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_painting + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake_painting).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
        print("Loss of disc this epoch: ", loss_disc)
        print("Loss of gen this epoch: ", loss_gen)

    return gen

def generate(photos, model):
    photos = torch.from_numpy(photos.copy())
    if torch.cuda.is_available():
        photos = photos.cuda()
    photos = photos.float()
    photos = photos.reshape(-1, 1, 3, 256, 256)
    for i in range(20):
        le_monet = model(photos[i])
        le_monet = le_monet.reshape(256, 256, 3)
        le_monet = le_monet.detach().cpu().numpy()
        print(le_monet.shape)
        #img = Image.fromarray(le_monet, 'RGB')
        ##img.show()
        imageio.imwrite("content/generated/%d.jpg" % (i,), le_monet.astype(np.uint8))