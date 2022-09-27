import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.util import montage

parser = argparse.ArgumentParser()
parser.add_argument('-rvs', help="Path to real_vs_synthetic directory", default='dir/path.csv')
parser.add_argument('-rvr', help="Path to real_vs_real directory", default='dir/path.csv')
parser.add_argument('-svs', help="Path to real_vs_real directory", default='dir/path.csv')
parser.add_argument('--outbox', '-ob', help="Path to save boxplot as", default='dir/boxplot.png')
opt = parser.parse_args()

with open("classes.txt") as f:
    classes = f.read().splitlines()

# get lpips files
rvs_files = [os.path.join(opt.rvs, p) for p in os.listdir(opt.rvs)]
rvr_files = [os.path.join(opt.rvr, p) for p in os.listdir(opt.rvr)]
svs_files = [os.path.join(opt.svs, p) for p in os.listdir(opt.svs)]

def get_all_classes(files):
    # Get RVS dataframe
    all_classes_df = pd.DataFrame(columns=["class", "sim"])
    for f in files:
        # get class name
        c = os.path.splitext(f)[0].split('lpips_')[-1]
        # get dataframe
        crows = pd.read_csv(f)
        # create new dataframe
        # new_df = pd.DataFrame({"class": [c]*len(crows), "sim": crows["sim"].tolist()})
        crows["class"] = [c]*len(crows)
        # append to big dataframe
        all_classes_df = pd.concat([all_classes_df, crows], axis=0, ignore_index=True)
    return all_classes_df

# get all classes dfs for all three cases
rvs_df = get_all_classes(rvs_files)
rvr_df = get_all_classes(rvr_files)
svs_df = get_all_classes(svs_files)

# print most similar cases
print("The most similar real vs real case has LPIPS of = ", rvr_df[rvr_df["path0"] != rvr_df["path1"]]["sim"].min())
print("The most similar synthetic vs synthetic case has LPIPS of = ", svs_df[svs_df["path0"] != svs_df["path1"]]["sim"].min())
print("The most similar real vs synthetic case has LPIPS of = ", rvs_df["sim"].min())

# save boxplots for metric
rvr_df["Type"] = ["RVR"]*len(rvr_df)
svs_df["Type"] = ["SVS"]*len(svs_df)
base_df = pd.concat([rvr_df, svs_df], ignore_index=True, axis=0)

plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
sns.boxplot(data=rvs_df, x="sim", y="class", order=classes)
plt.xlim(0, 1)
plt.xlabel("LPIPS Similarity")
plt.subplot(1, 2, 2)
sns.boxplot(data=base_df, x="sim", y="class", hue="Type", order=classes)
plt.xlim(0, 1)
plt.xlabel("LPIPS Similarity")
plt.savefig(opt.outbox, bbox_inches="tight")
