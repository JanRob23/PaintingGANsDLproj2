import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader
from utils import *
from fileIO import *
from cycleGan import *

def go(monet, photos):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(719)
    img_ds = ImageDataset(monet, photos)
    img_dl = DataLoader(img_ds, batch_size=5, pin_memory=True)
    photo_img, monet_img = next(iter(img_dl))

    gan = CycleGAN(3, 3, 20, device)

    save_dict = {
        'epoch': 0,
        'gen_mtp': gan.gen_mtp.state_dict(),
        'gen_ptm': gan.gen_ptm.state_dict(),
        'desc_m': gan.desc_m.state_dict(),
        'desc_p': gan.desc_p.state_dict(),
        'optimizer_gen': gan.RMSprop_gen.state_dict(),
        'optimizer_desc': gan.RMSprop_desc.state_dict()
    }
    save_checkpoint(save_dict, 'init.ckpt')

    gan.train(img_dl)

    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.plot(gan.gen_stats.losses, 'r', label='Generator Loss')
    plt.plot(gan.desc_stats.losses, 'b', label='Descriminator Loss')
    plt.legend()
    plt.show()

    _, ax = plt.subplots(5, 2, figsize=(12, 12))
    for i in range(5):
        photo_img, _ = next(iter(img_dl))
        pred_monet = gan.gen_ptm(photo_img.to(device)).cpu().detach()
        photo_img = unnorm(photo_img)
        pred_monet = unnorm(pred_monet)
        
        ax[i, 0].imshow(photo_img[0].permute(1, 2, 0))
        ax[i, 1].imshow(pred_monet[0].permute(1, 2, 0))
        ax[i, 0].set_title("Input Photo")
        ax[i, 1].set_title("Monet-esque Photo")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
    plt.show()

    ph_ds = PhotoDataset('/content/data/photos')
    ph_dl = DataLoader(ph_ds, batch_size=1, pin_memory=True)

    trans = transforms.ToPILImage()
    ###Save images
    if os.path.isfile('current.ckpt'):
        if device == 'cpu':
            t = cputqdm(ph_dl, leave=False, total=ph_dl.__len__())
        else:
            t = tqdm(ph_dl, leave=False, total=ph_dl.__len__())


    for i, photo in enumerate(t):
        with torch.no_grad():
            pred_monet = gan.gen_ptm(photo.to(device)).cpu().detach()
        pred_monet = unnorm(pred_monet)
        img = trans(pred_monet[0]).convert("RGB")
        img.save('content/PaintingGANs_DL_proj2/PanosCycleGan/customMonet/' + str(i + 1) + '.jpg')

if __name__ == "__main__":
    monet = 'C:/Users/Panos/Desktop/DLgansproject/Data/DatasetCycleGAN/augs'
    photos = 'C:/Users/Panos/Desktop/DLgansproject/Data/DatasetCycleGAN/photo_jpg'
    go(monet, photos)