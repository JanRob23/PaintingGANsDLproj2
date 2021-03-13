import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
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
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


class Gan(nn.Module):
    def __init__(self):
        super().__init__()
    # Hyperparameters etc.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = 3e-4
        self.z_dim = 64
        self.image_dim = 28 * 28 * 1  # 784
        self.batch_size = 32
        self.num_epochs = 50

    def train(self):
        disc = Discriminator(self.image_dim).to(self.device)
        gen = Generator(self.z_dim, self.image_dim).to(self.device)
        fixed_noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
        )
        datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]
        dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        opt_disc = optim.Adam(disc.parameters(), lr=self.lr)
        opt_gen = optim.Adam(gen.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        writer_fake = SummaryWriter(f"logs/fake")
        writer_real = SummaryWriter(f"logs/real")
        step = 0

        for epoch in range(self.num_epochs):
            for batch_idx, (real, _) in enumerate(loader):
                real = real.view(-1, 784).to(self.device)
                batch_size = real.shape[0]

                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                noise = torch.randn(batch_size, self.z_dim).to(self.device)
                fake = gen(noise)
                disc_real = disc(real).view(-1)
                lossD_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = disc(fake).view(-1)
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2
                disc.zero_grad()
                lossD.backward(retain_graph=True)
                opt_disc.step()

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                # where the second option of maximizing doesn't suffer from
                # saturating gradients
                output = disc(fake).view(-1)
                lossG = criterion(output, torch.ones_like(output))
                gen.zero_grad()
                lossG.backward()
                opt_gen.step()

                if batch_idx == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(loader)} \
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







            
