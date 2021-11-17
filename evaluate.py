# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from tqdm import tqdm
from utils.data_utils import ImageDataset
from torch.utils.data import DataLoader
from utils.utils import load_config, set_seed
from torchvision import transforms

# Set device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ensures reproducibility
set_seed(1399)

# ========================
# Load configs
# ========================

CONFIG_PATH = "configs/evaluate_configs.yaml"
config = load_config(CONFIG_PATH)
real_data_file = config["real_data_file"]
synthetic_data_file = config["synthetic_data_file"]
filenames_col = config["filenames_col"]
labels_col = config["labels_col"]
classes = config["classes"]
compute_similarity = config["similarity_check"]["compute"]
similarity_metric = config["similarity_check"]["similarity_metric"]
save_most_similar = config["similarity_check"]["save_most_similar"]
save_most_different = config["similarity_check"]["save_most_different"]
compute_quality = config["quality_check"]["compute"]
fid_imagenet = config["quality_check"]["fid_imagenet"]
fid_eye2gene = config["quality_check"]["fid_eye2gene"]
class_preds_eye2gene = config["quality_check"]["class_preds_eye2gene"]
verbose = True

image_transforms = []
# image resizing
if config['transformations']['resize_dim'] is not None:
    resize_dim = config['transformations']['resize_dim']
    image_transforms.append(transforms.Resize((resize_dim, resize_dim)))
# grayscale image conversion
if config['transformations']['grayscale']:
    image_transforms.append(transforms.Grayscale())
# compulsory - transformation to torch tensor
image_transforms.append(transforms.ToTensor())

# ========================
# Begin Evaluation
# ========================

# save results here
save_dir = os.path.join(synthetic_data_file.replace(os.path.basename(synthetic_data_file), ''), "metrics")

if compute_similarity:

    os.makedirs(os.path.join(save_dir, similarity_metric), exist_ok=True)

    def save_image(img_pairs, title, save_name):
        cols = ['Generated Images', 'Real Images']
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(6, 12))
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        axes = axes.ravel()
        for index, row in img_pairs.iterrows():
            synthetic_image = synthetic_data[int(row["gen_image_index"])][2]
            real_image = real_data[int(row["real_image_index"])][2]
            axes[2 * index].imshow(np.uint8(synthetic_image.squeeze() * 255), cmap=plt.cm.gray)
            axes[2 * index].axis('off')
            axes[2 * index + 1].imshow(np.uint8(real_image.squeeze() * 255), cmap=plt.cm.gray)
            axes[2 * index + 1].axis('off')
        plt.suptitle(title)
        fig.tight_layout()
        plt.savefig(save_name)
        plt.close()

    from utils.evaluate import calc_img_similarity_v2

    if verbose:
        print("Computing similarity between generated and real dataset...")

    # dictionaries for saving results
    sim_scores_per_class = {gene: None for gene in classes}

    for i, c in enumerate(classes):
        if verbose:
            print("Scoring class {}".format(c))

        real_data = ImageDataset(real_data_file, filenames_col, labels_col, [c], transforms.Compose(image_transforms))
        real_dataloader = DataLoader(real_data, batch_size=1024, shuffle=False, num_workers=8, drop_last=True)
        synthetic_data = ImageDataset(synthetic_data_file, filenames_col, labels_col, [c], transforms.Compose(image_transforms))
        synthetic_dataloader = DataLoader(synthetic_data, batch_size=50, shuffle=False, num_workers=8)

        # pass real images with generated images into the image similarity function
        sim_scores_per_class[c] = calc_img_similarity_v2(synthetic_dataloader, real_dataloader, similarity_metric)

        if save_most_similar:
            most_similar_images = sim_scores_per_class[c].head(5).reset_index(drop=True)
            save_as = os.path.join(save_dir, similarity_metric, "{}_most_similar.jpg".format(c))
            title = "5 most similar pairs out of {} pairs".format(len(sim_scores_per_class[c]))
            save_image(most_similar_images, title, save_as)
            # save metric values
            save_as = os.path.join(save_dir, similarity_metric, "{}_most_similar_metric_values.csv".format(c))
            most_similar_images.to_csv(save_as)

        if save_most_different:
            most_different_images = sim_scores_per_class[c].tail(5).reset_index(drop=True)
            save_as = os.path.join(save_dir, similarity_metric, "{}_most_different.jpg".format(c))
            title = "5 most different pairs out of {} pairs".format(len(sim_scores_per_class[c]))
            save_image(most_different_images, title, save_as)
            # save metric values
            save_as = os.path.join(save_dir, similarity_metric, "{}_most_different_metric_values.csv".format(c))
            most_different_images.to_csv(save_as)

    # save a single plot of histograms
    plt.figure(figsize=(15, 6))
    similarity_scores_df = pd.DataFrame(dict([(k, v[similarity_metric]) for k, v in sim_scores_per_class.items()]))
    similarity_scores_df.describe().to_csv(os.path.join(save_dir, similarity_metric, "summary.csv"))
    similarity_scores_df = pd.melt(similarity_scores_df, value_vars=classes).dropna()
    sns.violinplot(data=similarity_scores_df, x="variable", y="value")
    plt.xlabel("Gene")
    plt.ylabel("Similarity score")
    plt.xticks(rotation=45)
    plt.title("Distribution of {}".format(similarity_metric))
    plt.savefig(os.path.join(save_dir, similarity_metric, "scores_hist.jpg"))

