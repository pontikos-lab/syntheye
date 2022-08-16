import argparse
import lpips
import pandas as pd
import torch

device = torch.device('cuda:2') if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='paths1.csv')
parser.add_argument('-p1','--path1', type=str, default='paths2.csv')
parser.add_argument('-c', '--class_', type=str, default=None)
parser.add_argument('-o', '--out', type=str, default='./lpips/scores.csv')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
opt = parser.parse_args()

df0 = pd.read_csv(opt.path0)
df1 = pd.read_csv(opt.path1)

df0 = df0[df0.gene == opt.class_] if opt.class_ is not None else df0
df1 = df1[df1.gene == opt.class_] if opt.class_ is not None else df1

## Initializing the model
loss_fn = lpips.LPIPS(net='vgg',version=opt.version)

if(opt.use_gpu):
	loss_fn.to(device)

df_scores = pd.DataFrame(columns=['path0', 'path1', 'sim'])

for i in range(len(df0)):
    for j in range(len(df1)):

        path0 = df0.iloc[i]["file.path"]
        path1 = df1.iloc[j]["file.path"]

        # Load images
        img0 = lpips.im2tensor(lpips.load_image(path0)) # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(path1))

        if(opt.use_gpu):
            img0 = img0.to(device)
            img1 = img1.to(device)

        # Compute distance
        dist01 = loss_fn.forward(img0, img1)

        row = pd.DataFrame({'path0': [path0], 'path1': [path1], 'sim': [dist01.detach().cpu().numpy().squeeze()]})
        df_scores = pd.concat([df_scores, row], ignore_index=True, axis=0)
        print('Distance: %.3f'%dist01)


df_scores.to_csv(opt.out, index=False)