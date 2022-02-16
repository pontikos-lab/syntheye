""" Train model architectures """

# import libraries
import os
import argparse
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from utils.data_utils import *
from utils.utils import *

# Set device to gpu if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# command line args - these are just extra checkpoints
parser = argparse.ArgumentParser()
parser.add_argument("--train_from_checkpoint", help="Path to latest checkpoint to resume model training", default=None)
parser.add_argument("--train_from_ema", help="Path to latest ema checkpoint to resume model training", default=None)
parser.add_argument("--logfile", help="Path to logfile in which to resume saving metrics", default=None)
parser.add_argument("--save_disc_weights", help="whether to save discriminator weights", action="store_false")
parser.add_argument("--verbose", help="Prints model logs during training", default=True)
parser.add_argument("--seed", help="Sets a seed during model training", default=1399)
args = parser.parse_args()

# ================================
# LOAD ALL CONFIGURATION VALUES
# ================================

# load configs file
CONFIG_PATH = "configs/train_configs.yaml"
config = load_config(CONFIG_PATH)

# data specific configs
train_data_directory = config['train_data_file']
test_data_directory = config['test_data_file']
filenames_col = config['filenames_col']
labels_col = config['labels_col']
train_classes = config['train_classes']
class_mapping = config['class_mapping']
transformations = config['transformations']
calc_fid = config['calc_fid']

# model I/O specific configs
model = config['model']
z_dim = config['z_dim']
im_resolution = config['output_im_resolution']
assert im_resolution <= 1024, "Cannot generate images larger than 1024!"
batch_size = config['batch_size']
save_dir = filename(config) if config["save_weights_as"] is None else config["save_weights_as"]

# training specific configs
train_configs = {"epochs": config['epochs'],
                 "loss_fn": config['loss_fn'],
                 "n_disc_updates": config['n_disc_updates'],
                 "gen_lr": float(config['gen_lr']),
                 "disc_lr": float(config['disc_lr']),
                 "beta1": float(config['beta1']),
                 "beta2": float(config['beta2']),
                 "display_step": config['display_step'],
                 "n_samples_to_generate": config['n_samples_to_generate'],
                 "save_checkpoint_steps": config['save_checkpoint_steps'],
                 "train_from_checkpoint": args.train_from_checkpoint,
                 "train_from_ema": args.train_from_ema,
                 "logfile":args.logfile,
                 "parallel": config['parallel'],
                 "device_ids": config['device_ids'],
                 "ema": True}

# other configs - specified through command line args
save_disc_weights = args.save_disc_weights
verbose = args.verbose
seed = args.seed

# Set seed for training reproducibility
set_seed(seed)

# ===================================
# CREATE DATASET AND DATALOADERS
# ===================================

if verbose:
    print("Creating Dataset...", end="")

# transform the image data
image_transforms = []

if transformations['crop'] is not None:
    crop_size = (1450, 1450)
    image_transforms.append(transforms.CenterCrop(crop_size))

# image resizing
if transformations['resize_dim'] is not None:
    resize_dim = transformations['resize_dim']
    image_transforms.append(transforms.Resize((resize_dim, resize_dim)))

# grayscale image conversion
if transformations['grayscale']:
    image_transforms.append(transforms.Grayscale())

# horizontal flips
if transformations['random_flip'] is not None:
    image_transforms.append(transforms.RandomHorizontalFlip(p=0.3))

# compulsory - transformation to torch tensor
image_transforms.append(transforms.ToTensor())

# image normalization - normalizes between -1 and 1
if transformations['normalize']:
    if transformations['grayscale']:
        image_transforms.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        image_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

# load as pytorch dataset
train_images = ImageDataset(data_file=train_data_directory,
                            fpath_col_name=filenames_col,
                            lbl_col_name=labels_col,
                            class_vals=train_classes,
                            transforms=transforms.Compose(image_transforms),
                            class_mapping=class_mapping)

test_images = ImageDataset(data_file=test_data_directory,
                           fpath_col_name=filenames_col,
                           lbl_col_name=labels_col,
                           class_vals=train_classes,
                           fold="test",
                           transforms=transforms.Compose(image_transforms),
                           class_mapping=class_mapping)

if verbose:
    print("Number of images: " + str(len(train_images)))
    if labels_col is not None:
        print("Number of classes: " + str(train_images.n_classes))

# create pytorch data_loader
train_dataloader = DataLoader(train_images, batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_images, batch_size, shuffle=False, num_workers=4, drop_last=True)
train_configs['n_classes'] = train_images.n_classes

# =================================
# LOAD MODEL
# =================================

if verbose:
    print("Setting up model...")

# Deep convolutional GAN (DCGAN)
if model == "dcgan":
    # run a slightly modified demo architecture if no dataset provided
    if data_directory == "demo":
        from models.dcgan import dcgan_demo
        gen = dcgan_demo.Generator(z_dim).to(device)
        disc = dcgan_demo.Discriminator().to(device)
    # use original dcgan architecture if dataset provided
    else:
        from models.dcgan import dcgan
        gen = dcgan.Generator(z_dim).to(device)
        disc = dcgan.Discriminator().to(device)

    gan_model = (gen, disc)

# Multiple-scale gradients GAN (MSG-GAN)
elif model == "msggan":
    from models.msggan import msggan
    depth = int(np.log2(im_resolution) - 1)
    mode = "grayscale" if transformations['grayscale'] else "rgb"

    # load the GAN model
    gan_model = msggan.MSG_GAN(depth=depth, latent_size=z_dim, mode=mode, use_ema=True, use_eql=True, ema_decay=0.999,
                               device=device, device_ids=train_configs['device_ids'], calc_fid=calc_fid)

# Conditional MSG-GAN v1
elif model == "cmsggan1":
    from models.msggan import conditional_msggan
    depth = int(np.log2(im_resolution) - 1)
    mode = "grayscale" if transformations['grayscale'] else "rgb"
    # load the GAN model
    gan_model = conditional_msggan.MSG_GAN(depth=depth, latent_size=z_dim, n_classes=train_images.n_classes,
                                           mode=mode, use_ema=True, use_eql=True, ema_decay=0.999,
                                           device=device, device_ids=train_configs['device_ids'], calc_fid=calc_fid)

# Conditional MSG-GAN v2
elif model == "cmsggan2":
    from models.cmsgganv2 import conditional_msggan
    depth = int(np.log2(im_resolution) - 1)
    mode = "grayscale" if transformations['grayscale'] else "rgb"
    # load the GAN model
    gan_model = conditional_msggan.MSG_GAN(depth=depth, latent_size=z_dim, n_classes=train_images.n_classes,
                                             mode=mode, use_ema=True, use_eql=True, ema_decay=0.999,
                                             device=device, device_ids=train_configs['device_ids'], calc_fid=calc_fid)

# auxillary classifier multi-scale GAN
elif model == "acgan":
    from models.acgan import acgan
    depth = int(np.log2(im_resolution) - 1)
    mode = "grayscale" if transformations['grayscale'] else "rgb"
    # load the GAN model
    gan_model = acgan.ACGAN(depth=depth, latent_size=z_dim, n_classes=train_images.n_classes,
                            mode=mode, use_ema=True, use_eql=True, ema_decay=0.999,
                            device=device, device_ids=train_configs['device_ids'], calc_fid=calc_fid)

elif model == "stylegan2":
    from models.stylegan2.gan import STYLEGAN2
    depth = int(np.log2(im_resolution) - 1)
    mode = "grayscale" if transformations["grayscale"] else "rgb"
    # load the GAN model
    gan_model = STYLEGAN2(depth=depth, latent_size=z_dim, n_classes=train_images.n_classes,
                          mode=mode, use_ema=True, use_eql=True, ema_decay=0.999,
                          device=device, device_ids=train_configs['device_ids'], calc_fid=calc_fid)

else:
    raise Exception("Unknown model architecture! Accepted choices are [dcgan, msggan, cmsggan1, cmsggan2]...")

# =========================
# BEGIN MODEL TRAINING
# =========================

if verbose:
    print("Begin Model Training...")

if model == "dcgan":
    from utils.train_utils.dcgan_wgan_train import train

elif model in ["msggan", "cmsggan1", "cmsggan2", "acgan", "stylegan2"]:
    from utils.train_utils.msggan_train import train

else:
    raise Exception("Unknown model architecture! Accepted choices are [dcgan, msggan, cmsggan1, cmsggan2]...")

gan_model = train(gan_model, train_dataloader, test_dataloader, train_configs, device=device, checkpoints_fname=save_dir)

if verbose:
    print("Training Completed!")