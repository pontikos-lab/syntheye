""" Generate new images using trained generator """

# import libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import helpers.data_utils
from helpers.data_utils import get_noise, show_tensor_images, get_one_hot_labels
from helpers.utils import load_config, set_seed

# Set device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================
# LOAD CONFIGS
# ==================================

# load configs file
CONFIG_PATH = "model_configs.yaml"
config = load_config(CONFIG_PATH)

# generator settings
model_name = config['model']
weights_path = config['weights_path']
z_dim = config['z_dim']
im_resolution = config['output_im_resolution']
classes = config["classes"] # selected classes to generate, can be "all" or a list of selected classes
n_samples = config['n_test_samples'] # generate n images of the selected classes

# save formats
save_grid = config['save_grid'] # grid of all images
save_individual = config['save_individual'] # save individual images in a directory
interpolate_classes = config['interpolate_classes']
verbose = 1

# Set seed for training
set_seed(1299)

# ========================
# Generate Images!
# ========================

if verbose:
    print("Generating sample images...")

if model_name == "dcgan":
    generated_images = None
elif model_name == "wgan-gp":
    generated_images = None
elif model_name == "msggan" or model_name == 'cmsggan':
    from generators.msggan_gen import generate

    dataset = helpers.data_utils.ImageDataset(config["data_file"], config["filenames_col"], config["labels_col"])
    class_mapping = dataset.class2idx
    # if "all" is passed in config file, get the list of all classes from the original training data directory
    if classes == "all":
        classes = dataset.classes

    # generate images
    generated_images = generate(z_dim, n_samples, im_resolution, True, classes, class_mapping, weights_path, device)

else:
    print("pass")

# ===================================
# LOAD MODEL ARCHITECTURE
# ===================================

# SETUP MODEL
# if verbose:
#     print("Setting up model...")

# DCGAN
# if model_name == "dcgan":
#     # # run demo architecture if no dataset provided
#     # if config['data_directory'] == "demo":
#     #     from models import dcgan_demo
#     #     gen = dcgan_demo.Generator(z_dim).to(device)
#     # # use normal dcgan architecture if dataset provided
#     # else:
#     #     from models import dcgan
#     #     gen = dcgan.Generator(z_dim).to(device)
#
# # WGAN-GP
# elif model_name == "wgan-gp":
#     # run demo architecture if no dataset provided
#     if config['data_directory'] == "demo":
#         from models import wgangp_demo
#         gen = wgangp_demo.Generator(z_dim).to(device)
#     # use normal wgangp architecture if dataset provided
#     else:
#         from models import wgangp
#         gen = wgangp.Generator(z_dim).to(device)
# # MSG-GAN
# elif model_name == "msggan":
#     from models.msggan import msggan
#     depth = int(np.log2(im_resolution) - 1)
#     mode = "grayscale" if config['transformations']['grayscale'] else "rgb"
#     # load the GAN model
#     gan_model = msggan.MSG_GAN(depth=depth, latent_size=z_dim, mode=mode, use_ema=True, use_eql=True, ema_decay=0.999,
#                                device=device)
#     gen = gan_model.gen_shadow
#
# # CONDITIONAL MSGGAN
# elif model_name == "cmsggan":
#     from models.msggan import conditional_msggan
#     depth = int(np.log2(im_resolution) - 1)
#     mode = "grayscale" if config['transformations']['grayscale'] else "rgb"
#     # load the class information from training dataset
#     dataset = helpers.data_utils.ImageDataset(config['images_directory'])
#     gene_classes = dataset.classes
#     class_idx = torch.arange(dataset.n_classes)
#     class_encodings = get_one_hot_labels(class_idx, dataset.n_classes)
#     # load the GAN model
#     gan_model = conditional_msggan.MSG_GAN(depth=depth,
#                                            latent_size=z_dim,
#                                            n_classes=dataset.n_classes,
#                                            mode=mode,
#                                            use_ema=True,
#                                            use_eql=True,
#                                            ema_decay=0.999,
#                                            device=device)
#     gen = gan_model.gen_shadow
# else:
#     print("Unknown model architecture! Accepted choices are [\"dcgan\", \"wgan-gp\", \"pggan\", \"msggan\"]...")

# ===============================
# LOAD MODEL WEIGHTS
# ===============================

# load generator weights
# state_dict = torch.load(weights_path)
# gen.load_state_dict(state_dict)
# gen.eval()

# ====================
# GENERATE SOME DATA!
# ====================

# generate noise vectors
# noise_input = get_noise(n_samples=n_samples, z_dim=z_dim, device=device)

# # generate images - only select the final examples which are the highest resolution images
# generated_images = gen(noise_input)


# if model_name == "msggan" or model_name == "cmsggan":
#     from models.msggan.msggan import Generator
#     generated_images = Generator.adjust_dynamic_range(generated_images[-1])

if verbose:
    print("Saving results...")

# make directory to save results in
results_dir = "results/"
save_dir = os.path.basename(os.path.normpath(weights_path))[:-4]
os.makedirs(results_dir+save_dir, exist_ok=True)

# save images as grid format
if save_grid:
    # save results as grid of images
    show_tensor_images(generated_images,
                       n_rows=n_samples,
                       save_path=results_dir+save_dir+"/samples_grid",
                       normalize=False)

# save individual images into directories
if save_individual:
    for i in range(len(generated_images)):
        # if classes are provided, create the subdirectory for each class
        if classes is not None and i % n_samples == 0:
            class_dir = classes[i // n_samples]
            os.makedirs(results_dir+save_dir+"/"+class_dir, exist_ok=True)

        # visualize image
        img = generated_images[i]
        plt.imshow(img.squeeze(), cmap=plt.cm.gray)

        if classes is not None:
            plt.savefig(results_dir + save_dir + "/" + class_dir + "/gen_img_{}".format(i + 1))
        else:
            plt.savefig(results_dir + save_dir + "/gen_img_{}".format(i + 1))
