""" Train model architectures """

# import libraries
import os
import argparse
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from helpers.data_utils import *
from helpers.utils import *

# Set device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# command line args - these are just extra checkpoints
parser = argparse.ArgumentParser()
parser.add_argument("--train_from_checkpoint", help="Path to latest checkpoint to resume model training", default=None)
parser.add_argument("--save_disc_weights", help="whether to save discriminator weights", action="store_false")
parser.add_argument("--verbose", help="Prints model logs during training", default=True)
parser.add_argument("--seed", help="Sets a seed during model training", default=1399)
args = parser.parse_args()

# ================================
# LOAD ALL CONFIGURATION VALUES
# ================================

# load configs file
CONFIG_PATH = "model_configs.yaml"
config = load_config(CONFIG_PATH)

# data specific configs
data_directory = config['data_directory']
transformations = config['transformations']

# model I/O specific configs
model = config['model']
z_dim = config['z_dim']
im_resolution = config['output_im_resolution']
assert im_resolution <= 1024, "Cannot generate images larger than 1024!"
batch_size = config['batch_size']

# training specific configs
train_configs = {"epochs": config['epochs'],
                 "loss_fn": config['loss_fn'],
                 "n_disc_updates": config['n_disc_updates'],
                 "lr": config['lr'],
                 "beta1": config['beta1'],
                 "beta2": config['beta2'],
                 "display_step": config['display_step'],
                 "n_samples_to_generate": config['n_samples_to_generate'],
                 "save_checkpoint_steps": config['save_checkpoint_steps'],
                 "train_from_checkpoint": args.train_from_checkpoint}

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

if data_directory != "demo":

    # transform the image data
    image_transforms = []

    # image resizing
    if transformations['resize_dim'] is not None:
        resize_dim = transformations['resize_dim']
        image_transforms.append(transforms.Resize((resize_dim, resize_dim)))

    # grayscale image conversion
    if transformations['grayscale']:
        image_transforms.append(transforms.Grayscale())

    # compulsory - transformation to torch tensor
    image_transforms.append(transforms.ToTensor())

    # image normalization - normalizes between -1 and 1
    if transformations['normalize']:
        if transformations['grayscale']:
            image_transforms.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            image_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    # load as pytorch dataset
    train_images = ImageDataset(images_directory=data_directory, transforms=transforms.Compose(image_transforms))

    if verbose:
        print("Number of images: " + str(len(train_images)))

    # create pytorch data_loader
    data_loader = DataLoader(train_images, batch_size, shuffle=True, num_workers=4)

# if dataset is "demo", we will use the mnist dataset
else:
    # transform image values to be between -1 and 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    # if MNIST folder already exists then don't re-download it!
    download = True if not os.path.exists("demo_datasets/MNIST/") else False
    data_loader = DataLoader(MNIST('.', download=download, transform=transform), batch_size=batch_size, shuffle=True)

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

# Wasserstein GAN with Gradient Penalty (WGAN-GP)
elif model == "wgan-gp":
    # run demo architecture if no dataset provided
    if data_directory == "demo":
        from models.wgan import wgangp_demo
        gen = wgangp_demo.Generator(z_dim).to(device)
        disc = wgangp_demo.Critic().to(device)
    # use normal wgangp architecture if dataset provided
    else:
        from models.wgan import wgangp
        gen = wgangp.Generator(z_dim).to(device)
        disc = wgangp.Critic().to(device)

    gan_model = (gen, disc)

# Progressive Growing GAN (PGGAN)
elif model == "pggan":
    from models.pggan import pggan
    # set training depth
    mode = "grayscale" if transformations['grayscale'] else "rgb"
    gen = pggan.Generator(z_dim, mode=mode).to(device)
    disc = pggan.Discriminator(mode=mode).to(device)

    gan_model = (gen, disc)

# Multiple-scale gradients GAN (MSG-GAN)
elif model == "msggan":
    from models.msggan import msggan
    depth = int(np.log2(im_resolution) - 1)
    mode = "grayscale" if transformations['grayscale'] else "rgb"

    # load the GAN model
    gan_model = msggan.MSG_GAN(depth=depth, latent_size=z_dim, mode=mode, use_ema=True, use_eql=True, ema_decay=0.999,
                               device=device)

# Conditional MSG-GAN
elif model == "cmsggan":
    pass

else:
    print("Unknown model architecture! Accepted choices are [\"dcgan\", \"wgan-gp\", \"pggan\", \"msggan\"]...")

# =========================
# BEGIN MODEL TRAINING
# =========================

if verbose:
    print("Begin Model Training...")

if model in ["dcgan", "wgan-gp"]:
    from trainers.dcgan_wgan_train import train

elif model == "pggan":
    from trainers.pggan_train import train

elif model == "msggan":
    from trainers.msggan_train import train

gan_model = train(gan_model, data_loader, train_configs, device=device, checkpoints_fname=filename(config))

if verbose:
    print("Training Completed!")

# ===========================
# SAVING MODEL
# ===========================

# logs are automatically saved in runs/ but if we don't want to save them then we can just delete the folder
if not config['save_tensorboard']:
    # delete the latest logs file
    files = os.listdir("runs/")
    files = [os.path.join("runs/", filename) for filename in files]
    latest_log = max(files, key=os.path.getctime)
    import shutil
    shutil.rmtree(latest_log)

# saving weights
if config['save_weights']:
    if verbose:
        print("Saving GAN weights...", end="")
    # create a weights directory if it doesn't exist
    if not os.path.exists("weights/"):
        os.mkdir("weights/")
    # save the weights
    if args.save_disc_weights:
        save_weights(config, gan_model.gen, gan_model.dis)
    else:
        save_weights(config, gan_model.gen)

# =========================
# END OF FILE
# =========================
