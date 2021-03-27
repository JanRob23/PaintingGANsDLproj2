import itertools
import time
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from utils import *
from fileIO import *

class Resblock(nn.Module):
    def __init__(self, in_features, use_dropout=True, dropout_ratio=0.5):
        super().__init__()
        layers = list()
        layers.append(nn.ReflectionPad2d(1))
        layers.append(Convlayer(in_features, in_features, 3, 1, False, use_pad=False))
        layers.append(nn.Dropout(dropout_ratio))
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(in_features, in_features, 3, 1, padding=0, bias=True))
        layers.append(nn.InstanceNorm2d(in_features))
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.res(x)
class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=6):
        super().__init__()
        model = list()
        model.append(nn.ReflectionPad2d(3))
        model.append(Convlayer(in_ch, 64, 7, 1, False, True, False))
        model.append(Convlayer(64, 128, 3, 2, False))
        model.append(Convlayer(128, 256, 3, 2, False))
        for _ in range(num_res_blocks):
            model.append(Resblock(256))
        model.append(Upsample(256, 128))
        model.append(Upsample(128, 64))
        model.append(nn.ReflectionPad2d(3))
        model.append(nn.Conv2d(64, out_ch, kernel_size=7, padding=0))
        model.append(nn.Tanh())

        self.gen = nn.Sequential(*model)

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, in_ch, num_layers=4):
        super().__init__()
        model = list()
        model.append(nn.Conv2d(in_ch, 64, 4, stride=2, padding=1))
        model.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, num_layers):
            in_chs = 64 * 2**(i-1)
            out_chs = in_chs * 2
            if i == num_layers -1:
                model.append(Convlayer(in_chs, out_chs, 4, 1))
            else:
                model.append(Convlayer(in_chs, out_chs, 4, 2))
        model.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
        self.disc = nn.Sequential(*model)

    def forward(self, x):
        return self.disc(x)

class WassersteinGANLoss(nn.Module):
    def __init__(self):
        super(WassersteinGANLoss, self).__init__()

    def __call__(self, fake, real=None, generator_loss=True):
        if generator_loss:
            wloss = -fake.mean()
        else:
            wloss = real.mean() - fake.mean()
        return wloss



def Upsample(in_ch, out_ch, use_dropout=True, dropout_ratio=0.5):
    if use_dropout:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.Dropout(dropout_ratio),
            nn.GELU()
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
        )

def Convlayer(in_ch, out_ch, kernel_size=3, stride=2, use_leaky=True, use_inst_norm=True, use_pad=True):
    if use_pad:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 1, bias=True)
    else:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 0, bias=True)

    if use_leaky:
        actv = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        actv = nn.GELU()

    if use_inst_norm:
        norm = nn.InstanceNorm2d(out_ch)
    else:
        norm = nn.BatchNorm2d(out_ch)

    return nn.Sequential(
        conv,
        norm,
        actv
    )


#class Grad_penalty(model, real, fake, device='cpu'):
    #BATCH_SIZE, C, H, W = real.shape
    #epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    #inter_images = real * epsilon + fake * (1 - epsilon)
    #mixed_scores = discriminator(inter_images)
    #gradient = torch.autograd.grad(inputs =  inter_images,
      #                             outputs = mixed_scores,
     #                              grad_outputs=torch.ones_like(mixed_scores),
     #                              create_graph = True,
     #                              retain_graph = True,
     #                              )[0]
    #gradient = gradient.view(gradient.shape[0],-1)
    #gradient_norm = gradient.norm(2,dim =1 )
    #gradient_penalty = torch.mean((gradient_norm -1)**2)
   # return gradient_penalty


class CycleGAN(object):
    def __init__(self, in_ch, out_ch, epochs, device, start_lr=2e-4, lmbda=10, idt_coef=0.5, decay_epoch=0):
        self.epochs = epochs
        self.decay_epoch = decay_epoch if decay_epoch > 0 else int(self.epochs/2)
        self.lmbda = lmbda
        self.idt_coef = idt_coef
        self.device = device
        self.gen_mtp = Generator(in_ch, out_ch)
        self.gen_ptm = Generator(in_ch, out_ch)
        self.desc_m = Discriminator(in_ch)
        self.desc_p = Discriminator(in_ch)
        self.init_models()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.RMSprop_gen = torch.optim.RMSprop(itertools.chain(self.gen_mtp.parameters(), self.gen_ptm.parameters()),
                                         lr = start_lr)
        self.RMSprop_desc = torch.optim.RMSprop(itertools.chain(self.desc_m.parameters(), self.desc_p.parameters()),
                                          lr=start_lr)
        self.sample_monet = sample_fake()
        self.sample_photo = sample_fake()
        gen_lr = lr_sched(self.decay_epoch, self.epochs)
        desc_lr = lr_sched(self.decay_epoch, self.epochs)
        self.gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.RMSprop_gen, gen_lr.step)
        self.desc_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.RMSprop_desc, desc_lr.step)
        self.gen_stats = AvgStats()
        self.desc_stats = AvgStats()
        self.WassLoss = WassersteinGANLoss()


    def init_models(self):
        init_weights(self.gen_mtp)
        init_weights(self.gen_ptm)
        init_weights(self.desc_m)
        init_weights(self.desc_p)
        self.gen_mtp = self.gen_mtp.to(self.device)
        self.gen_ptm = self.gen_ptm.to(self.device)
        self.desc_m = self.desc_m.to(self.device)
        self.desc_p = self.desc_p.to(self.device)
        
    def train(self, photo_dl):
        for epoch in range(self.epochs):
            start_time = time.time()
            avg_gen_loss = 0.0
            avg_desc_loss = 0.0
            t = tqdm(photo_dl, leave=False, total=photo_dl.__len__())
            for i, (photo_real, monet_real) in enumerate(t):
                photo_img, monet_img = photo_real.to(self.device), monet_real.to(self.device)
                update_req_grad([self.desc_m, self.desc_p], False)
                self.RMSprop_gen.zero_grad()

                #Forward pass through generator
                fake_photo = self.gen_mtp(monet_img)
                fake_monet = self.gen_ptm(photo_img)

                cycl_monet = self.gen_ptm(fake_photo) #
                cycl_photo = self.gen_mtp(fake_monet) #

                id_monet = self.gen_ptm(monet_img)
                id_photo = self.gen_mtp(photo_img)

                # generator losses - identity, Adversarial, cycle consistency
                #cycle consistency loss is left as Standard GAN

                idt_loss_monet = self.l1_loss(id_monet, monet_img) * self.lmbda * self.idt_coef
                idt_loss_photo = self.l1_loss(id_photo, photo_img) * self.lmbda * self.idt_coef

                cycle_loss_monet =  self.l1_loss(cycl_monet, monet_img) * 20
                cycle_loss_photo =  self.l1_loss(cycl_photo, photo_img) * 20

                monet_desc = self.desc_m(fake_monet)
                photo_desc = self.desc_p(fake_photo)

                real = torch.ones(monet_desc.size()).to(self.device)

                #####Standard adv losses
                #adv_loss_monet = self.mse_loss(monet_desc, real)
                #adv_loss_photo = self.mse_loss(photo_desc, real)

                #######WASSENSTEIN GAN LOSSES!
                adv_loss_monet = self.WassLoss(monet_desc,real=None,generator_loss=True)
                adv_loss_photo = self.WassLoss(photo_desc,real=None,generator_loss=True)

                # total generator loss
                total_gen_loss = cycle_loss_monet + adv_loss_monet + cycle_loss_photo + adv_loss_photo + idt_loss_monet + idt_loss_photo
                
                avg_gen_loss += total_gen_loss.item()

                # backward pass
                total_gen_loss.backward()
                self.RMSprop_gen.step()

                # Forward pass through Descriminator
                #train iteration between Discriminators and Generators = 5:1
                for i in range (0,5):
                    update_req_grad([self.desc_m, self.desc_p], True)
                    self.RMSprop_desc.zero_grad()

                    fake_monet = self.sample_monet([fake_monet.cpu().data.numpy()])[0]
                    fake_photo = self.sample_photo([fake_photo.cpu().data.numpy()])[0]
                    fake_monet = torch.tensor(fake_monet).to(self.device)
                    fake_photo = torch.tensor(fake_photo).to(self.device)

                    monet_desc_real = self.desc_m(monet_img)
                    monet_desc_fake = self.desc_m(fake_monet)
                    photo_desc_real = self.desc_p(photo_img)
                    photo_desc_fake = self.desc_p(fake_photo)

                    real = torch.ones(monet_desc_real.size()).to(self.device)
                    fake = torch.ones(monet_desc_fake.size()).to(self.device)

                # Descriminator losses
                # --------------------

                #modify to wassenstein gan loss

                #monet_desc_real_loss = self.mse_loss(monet_desc_real, real)
                #monet_desc_fake_loss = self.mse_loss(monet_desc_fake, fake)
                #photo_desc_real_loss = self.mse_loss(photo_desc_real, real)
                #photo_desc_fake_loss = self.mse_loss(photo_desc_fake, fake)

                #Wassenstein loss for critics (+GRADIENT PENALTY)
                   # lambda =10
                   # monet_gradient_pen = Grad_penalty(CycleGAN,monet_desc_real,monet_desc_fake , device='cuda')
                   # photo_gradient_pen = Grad_penalty(CycleGAN, photo_desc_real, photo_desc_fake, device='cuda')

                    monet_desc_loss = self.WassLoss(monet_desc_fake,monet_desc_real,generator_loss = False)/2

                    photo_desc_loss = self.WassLoss(photo_desc_fake,photo_desc_real,generator_loss = False)/2


                #monet_desc_loss = (monet_desc_real_loss + monet_desc_fake_loss) / 2
                #photo_desc_loss = (photo_desc_real_loss + photo_desc_fake_loss) / 2
                    total_desc_loss = - monet_desc_loss - photo_desc_loss
                    avg_desc_loss += total_desc_loss.item()
                # Backward
                #Weight clip value : play around with it :)
                    clip= 0.01
                    monet_desc_loss.backward()
                    photo_desc_loss.backward()
                    self.RMSprop_desc.step()
                #Clip both discriminator's weight to the given clipping value
                    for p in self.desc_m.parameters():
                        p.data.clamp_(-clip,clip)
                    for p in self.desc_p.parameters():
                        p.data.clamp_(-clip, clip)


                t.set_postfix(gen_loss=total_gen_loss.item(), desc_loss=total_desc_loss.item())

            save_dict = {
                'epoch': epoch+1,
                'gen_mtp': self.gen_mtp.state_dict(),
                'gen_ptm': self.gen_ptm.state_dict(),
                'desc_m': self.desc_m.state_dict(),
                'desc_p': self.desc_p.state_dict(),
                'optimizer_gen': self.RMSprop_gen.state_dict(),
                'optimizer_desc': self.RMSprop_desc.state_dict()
            }
            save_checkpoint(save_dict, 'current.ckpt')
            
            avg_gen_loss /= photo_dl.__len__()
            avg_desc_loss /= photo_dl.__len__()
            time_req = time.time() - start_time
            
            self.gen_stats.append(avg_gen_loss, time_req)
            self.desc_stats.append(avg_desc_loss, time_req)
            
            print("Epoch: (%d) | Generator Loss:%f | Discriminator Loss:%f" % 
                                                (epoch+1, avg_gen_loss, avg_desc_loss))
      
            self.gen_lr_sched.step()
            self.desc_lr_sched.step()

