import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os

folders = ['Photos', 'cycleGAN', 'WcycleGAN', 'WcycleGANgp' ,'wcGANAuto', 'wcGANgpAuto', 'FastAuto', 'TransferNetAE']

_, ax = plt.subplots(4, len(folders),sharex=True, sharey=True, figsize=(20, 20))

for j in range(len(folders)):
    for i in range(1, 5):
        print(i)
        image = Image.open(f'Data/testResults/{folders[j]}/{i}.jpg')
        ax[i-1, j].imshow(image)
        if i == 1:
            ax[i-1, j].set_title(folders[j])
plt.show()

# for i in range(5):
#     photo_img, _ = next(iter(img_dl))
#     pred_monet = gan.gen_ptm(photo_img.to(device)).cpu().detach()
#     photo_img = unnorm(photo_img)
#     pred_monet = unnorm(pred_monet)
    
#     ax[i, 0].imshow(photo_img[0].permute(1, 2, 0))
#     ax[i, 1].imshow(pred_monet[0].permute(1, 2, 0))
#     ax[i, 0].set_title("Input Photo")
#     ax[i, 1].set_title("Monet-esque Photo")
#     ax[i, 0].axis("off")
#     ax[i, 1].axis("off")
# plt.show()
