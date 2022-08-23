import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.util import montage

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d', help="Path to directory", default='dir/path.csv')
parser.add_argument('--top', '-t', help="Save top T most similar iamges", default=1)
parser.add_argument('--outim', '-oi', help="Path to save similar pairs grid as", default='dir/pairs.png')
opt = parser.parse_args()

with open("classes.txt") as f:
    classes = f.read().splitlines()

# get lpips files
lpips_files = [os.path.join(opt.dir, p) for p in os.listdir(opt.dir)]

# save most similar image pairs
image_pairs = []
for f in lpips_files:
    # get class name
    c = f.split("lpips_")[1].split(".csv")[0]
    # get dataframe
    df = pd.read_csv(f)
    # sort dataframe by similarity value and select top T
    top_T = df.sort_values(by=["sim"], ascending=True).iloc[:opt.top]
    # append most similar image pairs
    img1 = Image.open(top_T["path0"].item()).convert('L')
    img1 = img1.resize((512, 512))
    img2 = Image.open(top_T["path1"].item()).convert('L')
    img2 = img2.resize((512, 512))
    image_pairs.append(np.array(img1))
    image_pairs.append(np.array(img2))
image_pairs = np.array(image_pairs)
# make grid
plt.figure(figsize=(10.66, len(lpips_files)*5.33))
grid = montage(arr_in=image_pairs, grid_shape=(len(lpips_files), 2), padding_width=1)
plt.axis('off')
plt.imshow(grid, plt.cm.gray)
plt.savefig(opt.outim, bbox_inches='tight')