import os, sys, argparse
from itertools import product
from tqdm import tqdm

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from lpips import load_image, im2tensor
from lpips.lpips import LPIPS

device = torch.device('cuda:3') if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='paths1.csv')
parser.add_argument('-p1','--path1', type=str, default=None)
parser.add_argument('-c', '--class_', type=str, default=None)
parser.add_argument('-o', '--out', type=str, default='./lpips_metric/scores.csv')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
opt = parser.parse_args()

class LPIPSDataset(Dataset):
    def __init__(self, path, class_):
        super(LPIPSDataset, self).__init__()
        self.path = path
        self.dataframe = pd.read_csv(self.path)
        self.class_ = class_
        self.files = list(self.dataframe[self.dataframe.gene == self.class_]["file.path"])
        self.len = len(self.files)

    def __getitem__(self, item):
        im = load_image(self.files[item])
        im = im2tensor(im)
        return self.files[item], im.squeeze()
    
    def __len__(self):
        return self.len

# load image dataset
dataset0 = LPIPSDataset(opt.path0, opt.class_)
dataset1 = LPIPSDataset(opt.path1, opt.class_) if opt.path1 is not None else LPIPSDataset(opt.path0, opt.class_)

# create dataloader
dataloader0 = DataLoader(dataset0, batch_size=16)
dataloader1 = DataLoader(dataset1, batch_size=16)

# Initializing the lpips calculator
metric = LPIPS(net='alex',version=opt.version)

# push to gpu
if(opt.use_gpu):
	metric.to(device)

# create and save a csv
df_scores = pd.DataFrame(columns=['path0', 'path1', 'sim'])
df_scores.to_csv(opt.out, index=False)

for xpath, x in tqdm(dataloader0):
    for ypath, y in dataloader1:
        if(opt.use_gpu):
            x = x.to(device)
            y = y.to(device)
        dm = metric.dist_matrix(x, y, normalize=True)
        pathslist = list(product(xpath, ypath))
        row = pd.DataFrame(pathslist, columns=["path0", "path1"])
        row["sim"] = dm.detach().cpu().numpy().ravel()
        row.to_csv(opt.out, header=False, index=False, mode='a')