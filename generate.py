""" Use a trained GAN model to synthesize new images """

# import libraries
import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from utils.data_utils import get_noise, get_one_hot_labels, combine_vectors
from utils.utils import load_config, set_seed

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
n_samples = config["n_samples"]
weights_dir = config["weights_dir"]
weights_path = config["weights_path"]
classes = config["classes"]
with open(config["class_mapping"]) as f:
    class_mapping = json.load(f)
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
    from models.msggan import conditional_msgganv2 as model

gen = model.MSG_GAN(depth=depth,
                    latent_size=z_dim,
                    n_classes=n_total_classes,
                    mode=mode,
                    use_ema=True,
                    use_eql=True,
                    ema_decay=0.999,
                    device=device).gen_shadow
gen = torch.nn.DataParallel(gen)
gen.load_state_dict(torch.load(os.path.join(weights_dir, weights_path)))
gen.eval()

# =====================
# Create GAN inputs
# =====================

noise_input = get_noise(n_samples, z_dim, device)
class_idxs = torch.tensor([class_mapping[c] for c in classes])
class_encoding = get_one_hot_labels(class_idxs, n_total_classes).to(device)

# ========================
# Generate synthetic data
# ========================

generated_images = torch.zeros(len(classes), n_samples, im_resolution, im_resolution)
for i in range(class_idxs.shape[0]):
    for j in range(noise_input.shape[0]):
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

results_dir = "results/"
save_dir = os.path.basename(os.path.normpath(weights_dir)) + "/"
os.makedirs(results_dir+save_dir+weights_path[:-4], exist_ok=True)
for c in classes:
    os.makedirs(os.path.join(results_dir, save_dir, weights_path[:-4], "generated_examples", c), exist_ok=True)


def save_image(img, save_as):
    img = Image.fromarray(np.uint8(img * 255), 'L')
    img.save(save_as, format='JPEG')


# create a dataframe of the filepaths and the labels for each synthetic image - useful for evaluation
filepaths_df = pd.DataFrame(columns=["file.path", "gene"])
# save images of each class
for i in range(len(classes)):
    for j in range(n_samples):
        with ThreadPoolExecutor() as executor:
            img = generated_images[i, j, :, :].numpy()
            save_as = results_dir + save_dir + weights_path[:-4] + "/generated_examples/" + classes[i] + "/gen_img_{}".format(j + 1)
            filepaths_df = filepaths_df.append({"file.path":save_as, "gene":classes[i]}, ignore_index=True)
            executor.submit(save_image, img, save_as)
filepaths_df.to_csv(os.path.join(results_dir, save_dir, weights_path[:-4], "generated_examples.csv"))
