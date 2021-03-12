import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_channels, features_disc):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels, features_disc, kernel_size=16, stride=2, padding=1), #122
            nn.LeakyReLU(0.2),
            self.block(features_disc, features_disc * 2, 12, 2, 0), # 56
            self.block(features_disc * 2, features_disc * 4, 12, 2, 1), # 24
            self.block(features_disc * 4, features_disc * 6, 12, 2, 1), # 8
            self.block(features_disc * 6, features_disc * 8, 5, 1, 0), # 4
            self.block(features_disc * 8, 1, 4, 1, 0), # 1
            nn.Sigmoid()
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, in_channels, features_gen):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # take 256 x 256 photo and downsample to 20 x 20
            self.block_downsample(in_channels, int(features_gen / 2), 12, 2, 1),    # 124 x 124
            self.block_downsample(int(features_gen / 2), features_gen, 10, 2, 0),  # 58 x 58
            self.block_downsample(features_gen, features_gen * 2, 9, 1, 0),  # 50 x 50
            self.block_downsample(features_gen * 2, features_gen * 3, 6, 2, 0),  # 23 x 23
            self.block_downsample(features_gen * 3, features_gen * 4, 4, 1, 0),  # 20 x 20
            # now upsample to 256 x 256 again with same parameters
            self.block_upsample(features_gen * 4, features_gen * 3, 4, 1, 0),
            self.block_upsample(features_gen * 3, features_gen * 2, 6, 2, 0),
            self.block_upsample(features_gen * 2, features_gen, 9, 1, 0),
            self.block_upsample(features_gen, int(features_gen / 2), 10, 2, 0),
            nn.ConvTranspose2d(int(features_gen / 2), in_channels, 12, 2, 1),
            nn.Tanh()
        )

    def block_downsample(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def block_upsample(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 256, 256
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    print("disc seems to work biatch")
    gen = Generator(in_channels, 8)
    #z = torch.randn((N, 3, 1, 1))
    assert gen(x).shape == (N, in_channels, H, W), "Generator test failed"
    print("done yay")
