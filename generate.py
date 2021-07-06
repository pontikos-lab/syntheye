""" Generate new images using trained generator """

# import libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from helpers.data_utils import get_noise, show_tensor_images
from helpers.utils import load_config, set_seed

# Set device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================
# LOAD CONFIGS
# ==================================

# load configs file
CONFIG_PATH = "model_configs.yaml"
config = load_config(CONFIG_PATH)
model_name = config['model']
z_dim = config['z_dim']
im_resolution = config['output_im_resolution']
n_samples = config['n_test_samples_to_generate']
weights_path = config['weights_path']
save_grid = config['save_grid']
verbose = 1

# Set seed for training
set_seed(1299)

# ===================================
# LOAD MODEL ARCHITECTURE
# ===================================

# SETUP MODEL
if verbose:
    print("Setting up model...")

# DCGAN
if model_name == "dcgan":
    # run demo architecture if no dataset provided
    if config['data_directory'] == "demo":
        from models import dcgan_demo
        gen = dcgan_demo.Generator(z_dim).to(device)
    # use normal dcgan architecture if dataset provided
    else:
        from models import dcgan
        gen = dcgan.Generator(z_dim).to(device)

# WGAN-GP
elif model_name == "wgan-gp":
    # run demo architecture if no dataset provided
    if config['data_directory'] == "demo":
        from models import wgangp_demo
        gen = wgangp_demo.Generator(z_dim).to(device)
    # use normal wgangp architecture if dataset provided
    else:
        from models import wgangp
        gen = wgangp.Generator(z_dim).to(device)

# MSG-GAN
elif model_name == "msggan":
    from models.msggan import msggan
    depth = int(np.log2(im_resolution) - 1)
    mode = "grayscale" if config['transformations']['grayscale'] else "rgb"
    # load the GAN model
    gan_model = msggan.MSG_GAN(depth=depth, latent_size=z_dim, mode=mode, use_ema=True, use_eql=True, ema_decay=0.999,
                               device=device)
    gen = gan_model.gen_shadow

else:
    print("Unknown model architecture! Accepted choices are [\"dcgan\", \"wgan-gp\", \"pggan\", \"msggan\"]...")

# ===============================
# LOAD MODEL WEIGHTS
# ===============================

if verbose:
    print("Generating sample images...")

# generate noise vectors
noise_input = get_noise(n_samples=n_samples, z_dim=z_dim, device=device)

# load generator weights
state_dict = torch.load(weights_path)
gen.load_state_dict(state_dict)
gen.eval()

# generate images - only select the final examples which are the highest resolution images
generated_images = gen(noise_input)
if model_name == "msggan":
    from models.msggan.msggan import Generator
    generated_images = Generator.adjust_dynamic_range(generated_images[-1])

# ================================
# SAVE GENERATOR OUTPUT
# ================================

if verbose:
    print("Saving results...")

# make a results directory if it doesnt exist
if not os.path.exists("results/"):
    os.mkdir("results/")

if not os.path.exists("results/"+os.path.basename(os.path.normpath(weights_path))[:-4]):
    os.mkdir("results/" + os.path.basename(os.path.normpath(weights_path))[:-4])

if save_grid:
    # save results as grid of images
    show_tensor_images(generated_images, n_rows=10,
                       save_path="results/"+os.path.basename(os.path.normpath(config["weights_path"]))[:-4]+"/samples_grid", normalize=False)
else:
    # save indivitual images
    for i in range(len(generated_images)):
        img = generated_images[i].detach().cpu()
        plt.imshow(img.squeeze(), cmap=plt.cm.gray)
        plt.savefig("results/"+os.path.basename(os.path.normpath(weights_path))[:-4]+"/gen_img_{}".format(i+1))
