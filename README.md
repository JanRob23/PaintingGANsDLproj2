# Project for the course Deep Learning by group 9
### Contributors
Charlie Albietz
Panagiotis Ritas
Jan-Henner Roberg

### Code used from
https://www.kaggle.com/c/gan-getting-started/data

https://github.com/eriklindernoren/Fast-Neural-Style-Transfer

## Describtion of repository
In this projct we used different deep learning techniques to transfer photos into Monet style paintings.

We used a CycleGAN, a CycleGAN using Wasserstein loss and a CycleGAN using Wasserstein loss with gradient clipping.

Additionally we used autoencoders either by training them on the output of the cycleGANs or by using content and style loss.

### Folders
Each Folder contains the models, dataloading and other utils/functions needed to run the specific model included in the folder. Each folder also contains a launch.py file which can be used to run the model in the specific folder. Note that the data path have to be adjusted to where the data is stored locally.

### Output

![image](https://user-images.githubusercontent.com/54030130/114086176-9bc65880-98b2-11eb-8f39-a216abf987c5.png)

