""" Use a trained GAN model to synthesize new images """

# import libraries
import os
import json
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from utils.data_utils import get_noise, get_one_hot_labels, combine_vectors
from utils.utils import load_config, set_seed
from tqdm import tqdm

# Set device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ensures reproducibility
set_seed(1399)

# =====================
# Load configs
# =====================

CONFIG_PATH = "configs/generate_configs.yaml"
config = load_config(CONFIG_PATH)
model_name = config["model_name"]
z_dim = config["z_dim"]
im_resolution = config["output_im_resolution"]
generate_randomly = config["generate_randomly"]
n_samples = config["n_samples"]
real_data_path = config["real_data_path"]
weights_dir = config["weights_dir"]
weights_path = config["weights_path"]
classes = config["classes"] # classes to specifically generate
if isinstance(classes, str):
    if classes.endswith(".txt"):
        with open(classes, 'r') as f:
            classes = f.read().splitlines()
with open(config["class_mapping"]) as f:
    class_mapping = json.load(f) # mapping from class to index
save_as = config["save_as"]
verbose = True

# ====================
# Load the GAN model
# ====================

if verbose:
    print("Generating synthetic data...")

n_total_classes= len(class_mapping)
depth = int(np.log2(im_resolution) - 1)
mode = "grayscale"

if model_name == "cmsggan1":
    from models.msggan import conditional_msggan as model
elif model_name == "cmsggan2":
    from models.cmsgganv2 import conditional_msggan as model

gen = model.MSG_GAN(depth=depth,
                    latent_size=z_dim,
                    n_classes=n_total_classes,
                    mode=mode,
                    use_ema=True,
                    use_eql=True,
                    ema_decay=0.999,
                    device=device).gen_shadow
gen = torch.nn.DataParallel(gen, device_ids=[device])
gen.load_state_dict(torch.load(os.path.join(weights_dir, weights_path), map_location=device))
gen.eval()

# ========================
# Generate synthetic data
# ========================

if n_samples == -1: # rebalance dataset (dynamically selects the number of images to create per class, based on largest class)

    # calculate number of synthetic images to make per class to rebalance dataset
    import pandas as pd
    real_data = pd.read_csv(real_data_path)
    real_data = real_data[real_data.gene.isin(classes)]
    classes2, class_sizes = np.unique(real_data.gene, return_counts=True)
    sizes_dict = dict(zip(classes2, class_sizes))
    largest_class = np.max(class_sizes)
    differences = {c:largest_class - sizes_dict[c] for c in classes}
    class_repeats = np.repeat(classes, list(differences.values()))

    # class_idxs = torch.tensor([class_mapping[c] for c in classes])
    # class_idxs = torch.repeat_interleave(class_idxs, torch.tensor(list(differences.values())))
    # class_encoding = get_one_hot_labels(class_idxs, n_total_classes).to(device)

    results_dir = "/home/zchayav/projects/syntheye/synthetic_datasets/"
    synth_dataset_path = os.path.join(results_dir, save_as)
    os.makedirs(synth_dataset_path, exist_ok=True)

    # make a subdirectory for storing images of each class
    for c in classes:
        os.makedirs(os.path.join(synth_dataset_path, c), exist_ok=True)

    def save_image(img, save_as):
        img = Image.fromarray(np.uint8(img * 255), 'L')
        img.save(save_as, format='JPEG')

    # create a dataframe of the filepaths and the labels for each synthetic image - useful for evaluation
    filepaths_df = pd.DataFrame(columns=["file.path", "gene", "noise_vector", "class_encoding"])

    for i, c in tqdm(enumerate(class_repeats)):
        noise_input = torch.randn(1, z_dim).to(device)
        noise_input = noise_input / noise_input.norm(dim=-1, keepdim=True) * (z_dim ** 0.5)
        class_idx = torch.tensor([class_mapping[c]]).to(device)
        class_encoding = get_one_hot_labels(class_idx, n_total_classes)
        latent = combine_vectors(noise_input, class_encoding)
        img = gen(latent)[-1].squeeze()
        # adjust image pixel values
        img = model.Generator().adjust_dynamic_range(img.detach().to('cpu'))
        img = img.numpy()

        with ThreadPoolExecutor() as executor:
            img_save_path = os.path.join(synth_dataset_path, class_repeats[i], "gen_img_{}.png".format(i+1))
            filepaths_df = filepaths_df.append({"file.path":img_save_path, "gene":c, "noise_vector": noise_input.cpu().numpy().tolist(), "class_encoding": class_encoding.cpu().numpy().tolist()}, ignore_index=True)
            executor.submit(save_image, img, img_save_path)
    
    filepaths_df.to_csv(os.path.join(synth_dataset_path, "generated_examples.csv"), index=False)


elif n_samples > 0:

    # initialize to array of zeros
    generated_images = torch.zeros(len(classes), n_samples, im_resolution, im_resolution)

    # each image in each class uses a randomly generated noise vector 
    # (as opposed to just generating from a single noise vector concatenated to each class label)
    if generate_randomly:

        # create new data - initialize to zeros array
        noise_input = torch.zeros(n_samples, len(classes), z_dim).to(device)
        class_idxs = torch.tensor([class_mapping[c] for c in classes])
        # cone hot encoding array of classes
        class_encoding = get_one_hot_labels(class_idxs, n_total_classes).to(device)

        # generate new images randomly
        for i in range(len(classes)):
            for j in range(noise_input.shape[0]):
                if model_name == "cmsggan1":
                    noise = get_noise(1, 512, device)
                    # normalize noise vector
                    noise = noise / noise.norm(dim=-1, keepdim=True) * (z_dim ** 0.5)
                    noise_input[j] = noise.squeeze()
                    latent_input = combine_vectors(noise, class_encoding[i].view(1, -1))
                elif model_name == "cmsggan2":
                    noise = get_noise(1, 512, device)
                    # normalize noise vector
                    noise = noise / noise.norm(dim=-1, keepdim=True) * (z_dim ** 0.5)
                    noise_input[j] = noise.squeeze()
                    latent_input = (noise, torch.tensor([class_idxs[i]]).to(device))
                else:
                    raise Exception("model name has to be cmsggan1 or cmsggan2!")
                with torch.no_grad():
                    try:
                        generated_images[i, j, :, :] = gen(*latent_input)[-1].squeeze()
                    except:
                        generated_images[i, j, :, :] = gen(latent_input)[-1].squeeze()

    else:
        # create new data
        noise_input = get_noise(n_samples, z_dim, device)
        noise_input = noise_input / noise_input.norm(dim=-1, keepdim=True) * (z_dim ** 0.5)
        class_idxs = torch.tensor([class_mapping[c] for c in classes])
        class_encoding = get_one_hot_labels(class_idxs, n_total_classes).to(device)

        # generate new images systematically
        for i in range(len(classes)):
            for j in range(n_samples):
                if model_name == "cmsggan1":
                    latent_input = combine_vectors(noise_input[j].view(1, -1), class_encoding[i].view(1, -1))
                elif model_name == "cmsggan2":
                    latent_input = (noise_input[j][None, :], torch.tensor([class_idxs[i]]).to(device))
                else:
                    raise Exception("model name has to be cmsggan1 or cmsggan2!")
                with torch.no_grad():
                    try:
                        generated_images[i, j, :, :] = gen(*latent_input)[-1].squeeze()
                    except:
                        generated_images[i, j, :, :] = gen(latent_input)[-1].squeeze()

    # adjust image pixel values
    generated_images = model.Generator().adjust_dynamic_range(generated_images.detach().to('cpu'))

    # =====================
    # Save synthetic data
    # =====================

    if verbose:
        print("Saving synthetic data...")

    results_dir = "/home/zchayav/projects/syntheye/synthetic_datasets/"
    synth_dataset_path = os.path.join(results_dir, save_as)
    os.makedirs(synth_dataset_path, exist_ok=True)

    # make a subdirectory for storing images of each class
    for c in classes:
        os.makedirs(os.path.join(synth_dataset_path, c), exist_ok=True)


    def save_image(img, save_as):
        img = Image.fromarray(np.uint8(img * 255), 'L')
        img.save(save_as, format='JPEG')


    # create a dataframe of the filepaths and the labels for each synthetic image - useful for evaluation
    filepaths_df = pd.DataFrame(columns=["file.path", "gene", "noise_vector", "class_encoding"])
    # save images of each class
    for i in range(len(classes)):
        for j in range(n_samples):
            with ThreadPoolExecutor() as executor:
                img = generated_images[i, j, :, :].numpy()
                img_save_path = os.path.join(synth_dataset_path, classes[i], "gen_img_{}.png".format(j+1))
                filepaths_df = filepaths_df.append({"file.path":img_save_path, "gene":classes[i], "noise_vector": noise_input[j].cpu().numpy().tolist(), "class_encoding": class_encoding[i].cpu().numpy().tolist()}, ignore_index=True)
                executor.submit(save_image, img, img_save_path)
    filepaths_df.to_csv(os.path.join(synth_dataset_path, "generated_examples.csv"), index=False)

else:
    pass
