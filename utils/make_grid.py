import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid
import torch

# figsize = (6, )
dstpath = "/home/zchayav/projects/syntheye/synthetic_datasets/all_folds/montage.png"
dirpath = "/home/zchayav/projects/syntheye/synthetic_datasets/stylegan2_synthetic_50perclass/generated_examples.csv"
df = pd.read_csv(dirpath)

with open("classes.txt") as f:
    genes = f.read().splitlines()[:9]
n_samples = 5

np.random.seed(2021)

images = torch.zeros(n_samples*len(genes), 1, 512, 512)

# plt.figure()
k = 0
for i, gene in enumerate(genes):
    sample_images = np.random.choice(df[df["gene"] == gene]["file.path"].tolist(), size=n_samples, replace=False)    
    for j in range(n_samples):
        # plt.subplot(len(genes), n_samples, i*n_samples + j + 1)
        img = Image.open(sample_images[j])
        img = np.array(img) / 255
        images[k, :, :, :] = torch.as_tensor(img)[None, :, :]
        k += 1

plt.figure(figsize=(12,24))
grid = make_grid(images, nrow=n_samples)
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.savefig(dstpath)
        # plt.imshow(img.resize((256, 256)), plt.cm.gray)
        # plt.axis("off")
# plt.tight_layout()
# plt.show()
# plt.savefig(dstpath)
# plt.close()
