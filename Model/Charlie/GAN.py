import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorflow.examples.tutorials.mnist import input_data

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim): 
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.gen(x)
    
class Gan():
    def __init__(self):
        super().__init__()
        #Hyperparameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = 0.0003
        self.z_dim = 64
        self.image_dim = 28 * 28 * 1
        self.batch = 32
        self.num_epochs = 50

    def train(self):
        disc = Discriminator(self.image_dim).to(self.device)
        gen = Generator(self.z_dim, self.image_dim).to(self.device)
        
        fixed_noise = torch.randn((self.batch, self.z_dim)).to(self.device)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]
        )
        dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
        loader = DataLoader(dataset, self.batch, shuffle=True)
        opt_disc = nn.Adam(disc.parameters(), lr=self.learning_rate)
        opt_gen = nn.Adam(gen.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        #tensor board
        writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
        writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
        step = 0

        for epoch in range(self.num_epochs):
            for batch_inx, (real, _) in enumerate(loader):
                real = real.view(-1, self.image_dim).to(self.device)
                batch_size = real.shape[0]

                #Train Discrinator
                noise = torch.randn((batch_size,self.z_dim)).to(self.device)
                fake = gen(noise)
                disc_real = disc(real).view(-1)
                lossD_real = criterion(disc_real, torch.ones_like(disc_real)) 
                disc_fake = disc(fake).view(-1)
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_fake + lossD_real) / 2
                disc.zero_grad()
                lossD.backward(retain_graph=True)
                opt_disc.step()

                #Train Generator
                output = disc(fake).view(-1)
                lossG = criterion(output, torch.ones_like(output))
                gen.zero_grad()
                lossG.backward()
                opt_gen.step()

                #TensorBoard
                if batch_inx == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_inx}/{len(loader)} \
                            Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                    )

                    with torch.no_grad():
                        fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                        data = real.reshape(-1, 1, 28, 28)
                        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                        writer_fake.add_image(
                            "Mnist Fake Images", img_grid_fake, global_step=step
                        )
                        writer_real.add_image(
                            "Mnist Real Images", img_grid_real, global_step=step
                        )
                        step += 1








        
