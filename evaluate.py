# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from PIL import Image
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
if classes == "all":
    with open("classes.txt") as f:
        classes = f.read().splitlines()
compute_similarity = config["similarity_check"]["compute"]
alpha, beta = config["similarity_check"]["process_images"]["alpha"], config["similarity_check"]["process_images"]["beta"]
filtering = config["similarity_check"]["process_images"]["filtering"]
kernel, ksize = filtering["kernel"], filtering["size"]
thresholding = config["similarity_check"]["process_images"]["thresholding"]
threshfunc, tsize = thresholding["function"], thresholding["size"]
similarity_metric = config["similarity_check"]["similarity_metric"]
save_most_similar = config["similarity_check"]["save_most_similar"]
save_most_different = config["similarity_check"]["save_most_different"]
compute_quality = config["quality_check"]["compute"]
quality_metric = config["quality_check"]["quality_metric"]
save_dir = config["save_dir"]
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
os.makedirs(save_dir, exist_ok=True)

if compute_similarity:

    def save_image(img_pairs, title, save_name):
        cols = ['Generated Images', 'Real Images']
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(6, 12))
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        axes = axes.ravel()
        for index, row in img_pairs.iterrows():
            synthetic_image = Image.open(row["gen_image_path"])
            real_image = Image.open(row["real_image_path"])
            axes[2 * index].imshow(synthetic_image, cmap=plt.cm.gray)
            axes[2 * index].axis('off')
            axes[2 * index + 1].imshow(real_image, cmap=plt.cm.gray)
            axes[2 * index + 1].axis('off')
        plt.suptitle(title)
        fig.tight_layout()
        plt.savefig(save_name)
        plt.close()

    from utils.evaluation_utils import ComputeSimilarity

    if verbose:
        print("Computing similarity between generated and real dataset...")

    # dictionaries for saving results
    sim_scores_per_class = {gene: None for gene in classes}

    for i, c in enumerate(classes):
        if verbose:
            print("Scoring class {}".format(c))

        real_data = ImageDataset(real_data_file, filenames_col, labels_col, [c], transforms.Compose(image_transforms))
        real_dataloader = DataLoader(real_data, batch_size=1024, shuffle=False, num_workers=8)
        synthetic_data = ImageDataset(synthetic_data_file, filenames_col, labels_col, [c], transforms.Compose(image_transforms))
        synthetic_dataloader = DataLoader(synthetic_data, batch_size=1024, shuffle=False, num_workers=8)

        # pass real images with generated images into the image similarity function
        sim_scores_per_class[c] = ComputeSimilarity(metric_name=similarity_metric)(synthetic_dataloader,
                                                                                   real_dataloader,
                                                                                   process_images=True,
                                                                                   alpha=alpha,
                                                                                   beta=beta,
                                                                                   filter=kernel,
                                                                                   fsize=ksize,
                                                                                   threshold=threshfunc,
                                                                                   tsize=tsize)

        # save distance matrix
        sim_scores_per_class[c].to_csv(os.path.join(save_dir, "{}_distance_matrix.csv".format(c)))

        if save_most_similar:
            most_similar_images = sim_scores_per_class[c].head(5).reset_index(drop=True)
            save_as = os.path.join(save_dir, "{}_most_similar.jpg".format(c))
            title = "5 most similar pairs out of {} pairs".format(len(sim_scores_per_class[c]))
            save_image(most_similar_images, title, save_as)
            # save metric values
            save_as = os.path.join(save_dir, similarity_metric, "{}_most_similar_metric_values.csv".format(c))
            # most_similar_images.to_csv(save_as)

        if save_most_different:
            most_different_images = sim_scores_per_class[c].tail(5).reset_index(drop=True)
            save_as = os.path.join(save_dir, "{}_most_different.jpg".format(c))
            title = "5 most different pairs out of {} pairs".format(len(sim_scores_per_class[c]))
            save_image(most_different_images, title, save_as)
            # save metric values
            save_as = os.path.join(save_dir, similarity_metric, "{}_most_different_metric_values.csv".format(c))
            # most_different_images.to_csv(save_as)

    # save a single plot of histograms
    plt.figure(figsize=(15, 6))
    similarity_scores_df = pd.DataFrame(dict([(k, v[similarity_metric]) for k, v in sim_scores_per_class.items()]))
    similarity_scores_df.describe().to_csv(os.path.join(save_dir, "summary.csv"))
    similarity_scores_df = pd.melt(similarity_scores_df, value_vars=classes)
    similarity_scores_df["variable"] = similarity_scores_df["variable"].astype("string")
    similarity_scores_df["value"] = similarity_scores_df["value"].astype("float")
    sns.violinplot(data=similarity_scores_df, x="variable", y="value")
    plt.xlabel("Gene")
    plt.ylabel("Similarity score")
    plt.xticks(rotation=45)
    plt.title("Distribution of {}".format(similarity_metric))
    plt.savefig(os.path.join(save_dir, "scores_hist.jpg"))

if compute_quality:

    from utils.evaluation_utils import ComputeQuality

    # dictionaries for saving results
    qual_scores_per_class = {gene: None for gene in classes}

    for i, c in enumerate(classes):
        if verbose:
            print("Scoring class {}".format(c))

        synthetic_data = ImageDataset(synthetic_data_file, filenames_col, labels_col, [c], transforms.Compose(image_transforms))
        synthetic_dataloader = DataLoader(synthetic_data, batch_size=50, shuffle=False, num_workers=8)

        # pass real images with generated images into the image similarity function
        qual_scores_per_class[c] = ComputeQuality(quality_metric=quality_metric)(synthetic_dataloader)

        # save quality metric
        qual_scores_per_class[c].to_csv(os.path.join(save_dir, "{}_quality_scores.csv".format(c)))
