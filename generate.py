""" Generates new images using trained generator and evaluates them """

# import libraries
import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader
from helpers.data_utils import get_noise, ImageDataset, show_tensor_images, get_one_hot_labels
from helpers.utils import load_config, set_seed
from torchvision import transforms

# Set device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================
# LOAD CONFIGS
# ==================================

# read configs file
CONFIG_PATH = "model_configs.yaml"
config = load_config(CONFIG_PATH)

# required generator settings
model_name = config['model']
weights_dir = config['weights_dir']
weights_path = config['weights_path']
z_dim = config['z_dim']
im_resolution = config['output_im_resolution']
classes = config["gen_classes"] # selected classes to generate, can be "all" or a list of selected classes
n_samples = config['n_test_samples'] # generate n images of the selected classes

# TESTING SETTINGS

# list of all image transforms - this will be needed for loading in the real images to compare with generated
transformations = config['transformations']
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
# image_transforms = transforms.Compose(image_transforms)

# 1. Metrics computation
calc_mutual_info = config['evaluate']['mutual_information']
calc_fid_imagenet = config['evaluate']['fid_imagenet']
calc_fid_eye2gene = config['evaluate']['fid_eye2gene']
class_preds_eye2gene = config['evaluate']['class_preds_eye2gene']

# 2. save metrics
save_individual = config['save_images']['as_individual'] # save individual images in a directory
save_most_similar = config['save_images']['most_similar']
save_most_different = config['save_images']['most_different']
verbose = 1

# Set seed for reproducibility
set_seed(1399)

# ============================================
# Load the real dataset (For testing purposes)
# ============================================

dataset = ImageDataset(config["data_file"], config["filenames_col"], config["labels_col"], config['train_classes'])
class_mapping = dataset.class2idx
# if "all" is passed in config file, get the list of all classes from the original training data directory
if classes == "all":
    # classes = dataset.classes
    cls, sizes = np.unique(dataset.img_labels, return_counts=True)
    classes = cls[np.argsort(sizes)][::-1]

# ========================
# Generate New Images!
# ========================

if verbose:
    print("Generating sample images...")

# TODO: Write generator scripts for dcgan and wgan
if model_name == "dcgan":
    generated_images = None

elif model_name == "wgan-gp":
    generated_images = None

elif model_name in ["msggan", "cmsggan", "cmsgganv2"]:

    # import generator function which uses model to generate images
    from generators.msggan_gen import generate
    # generate images
    generated_images = generate(z_dim,
                                n_samples,
                                im_resolution,
                                True,
                                classes,
                                class_mapping,
                                False,
                                weights_dir+weights_path,
                                model_name,
                                device)

else:
    raise ValueError("Generator can be of following types only: dcgan, wgangp, msggan, biggan.")

# ==========================
# EVALUATE IMAGES
# ==========================

# 1. Check that images are not just memorized training set examples using mutual information metric. This will compute a
# distance matrix between each generated image G_i of class C with every real image of class C.
if calc_mutual_info:
    from helpers.evaluate import mutual_information
    if verbose:
        print("Computing Mutual Information between generated and real dataset...")
    if classes is not None:
        # dictionaries for saving results
        MI_scores_per_class = {gene: None for gene in classes}
        if save_most_similar:
            top5_similar_images_per_class = {gene: None for gene in classes}
        if save_most_different:
            top5_different_images_per_class = {gene: None for gene in classes}

        for i, c in enumerate(classes):
            print("Scoring class {}".format(c))
            # create a data loader for the real images of the specific class
            class_dataset = ImageDataset(config["data_file"], config["filenames_col"], config["labels_col"],
                                         class_vals=[c], transforms=transforms.Compose(image_transforms))
            batch_size = 1024 if len(dataset) > 1024 else len(dataset)
            class_dataloader = DataLoader(class_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

            # pass real images with generated images into mutual info function
            MI_scores_per_class[c], most_similar, most_different = \
                mutual_information(generated_images[i], class_dataloader, save_most_similar, save_most_different)

            if save_most_similar:
                top5_similar_images_per_class[c] = most_similar
            if save_most_different:
                top5_different_images_per_class[c] = most_different
    else:
        pass

# 2. Calculate Frechet Inception Distance
if calc_fid_imagenet:
    from helpers.evaluate import compute_fid
    if verbose:
        print("Computing Frechet Inception Distance between generated and real images...")
    # compute FID using pretrained imagenet model
    if classes is not None:
        fid_imagenet_per_class = {gene: None for gene in classes}
        for i, c in enumerate(classes):
            class_dataset = ImageDataset(config["data_file"], config["filenames_col"], config["labels_col"],
                                         class_vals=[c], transforms=transforms.Compose(image_transforms[:-1]))
            real_images = class_dataset.get_samples(n_samples)
            fid_imagenet_per_class[c] = compute_fid(generated_images[i], real_images, device=device)
    else:
        pass

# 3. Calculate FID using Eye2Gene model
if calc_fid_eye2gene:
    from helpers.evaluate import compute_fid_eye2gene
    if verbose:
        print("Computing Frechet Inception Distance (use pretrained eye2gene model) "
              "between generated and real images...")
    # compute FID using eye2gene model
    if classes is not None:
        fid_eye2gene_per_class = {gene: None for gene in classes}
        for i, c in enumerate(classes):
            class_dataset = ImageDataset(config["data_file"], config["filenames_col"], config["labels_col"],
                                         class_vals=[c], transforms=transforms.Compose(image_transforms[:-1]))
            # Grab n examples of real images from the specific class
            real_images = class_dataset.get_samples(n_samples)
            fid_eye2gene_per_class[c] = compute_fid_eye2gene(generated_images[i]*255, real_images*255)

    else:
        pass

# 4. Create confusion matrix using Eye2Gene model
if class_preds_eye2gene:
    from helpers.evaluate import compute_class_confidence
    if verbose:
        print("Creating confusion matrix for generated images...")
    # predict on generated images
    predictions = []
    assert classes is not None, "Can make predictions only for images of specific classes!"
    for i, c in enumerate(classes):
        images = generated_images[i, :, :, :].numpy()*255
        predictions.append(compute_class_confidence(images))
    predictions = pd.concat([predictions[i].idxmax(axis=1) for i in range(len(predictions))], axis=1)
    # create confusion matrix
    confusions = pd.DataFrame(0, index=classes, columns=classes)
    for i, col in enumerate(confusions.columns):
        confusions[col] = confusions[col].combine(predictions[i].value_counts(normalize=0), max)

# ===========================
# SAVE RESULTS
# ===========================

if verbose:
    print("Saving results...")

# Create results directory
results_dir = "results/"
save_dir = os.path.basename(os.path.normpath(weights_dir)) + "/"
os.makedirs(results_dir+save_dir+weights_path[:-4], exist_ok=True)

# save individual images into directories
if save_individual:
    os.makedirs(results_dir + save_dir + weights_path[:-4] + "/generated_examples/", exist_ok=True)
    # make subdirectories for every class
    if classes is not None:
        for gene in classes:
            os.makedirs(results_dir+save_dir+weights_path[:-4]+"/generated_examples/"+gene, exist_ok=True)

    def save_image(img, save_as):
        img = Image.fromarray(np.uint8(img * 255), 'L')
        img.save(save_as, format='JPEG')

    # save images of each class
    if classes is not None:
        for i in range(len(classes)):
            for j in range(n_samples):
                with ThreadPoolExecutor() as executor:
                    img = generated_images[i, j, :, :].numpy()
                    save_as = results_dir + save_dir + weights_path[:-4] + "/generated_examples/" + classes[i] + "/gen_img_{}".format(j+1)
                    executor.submit(save_image, img, save_as)
    else:
        pass

# save histograms of the mutual information scores
if calc_mutual_info:
    # make directory for storing metrics
    os.makedirs(results_dir + save_dir + weights_path[:-4] + "/metrics/mutual_info_scores/", exist_ok=True)

    def save_image(img_pairs, save_as):
        cols = ['Generated Images', 'Real Images']
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(6, 12))
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        axes = axes.ravel()
        for j, pair in enumerate(img_pairs):
            axes[2 * j].imshow(pair[0].squeeze(), cmap=plt.cm.gray)
            axes[2 * j].axis('off')
            axes[2 * j + 1].imshow(pair[1].squeeze(), cmap=plt.cm.gray)
            axes[2 * j + 1].axis('off')
        fig.tight_layout()
        plt.savefig(save_as)
        plt.close()

    if classes is not None:
        # save a single plot of histograms
        plt.figure(figsize=(15, 6))
        MI_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in MI_scores_per_class.items()]))
        MI_df.describe().to_csv(results_dir+save_dir+weights_path[:-4]+"/metrics/mutual_info_scores/summary.csv")
        MI_df = pd.melt(MI_df, value_vars=classes).dropna()
        # print(MI_df)
        sns.violinplot(data=MI_df, x="variable", y="value")
        plt.xlabel("Gene")
        plt.ylabel("MI score")
        plt.xticks(rotation=45)
        plt.title("Distribution of Mutual Information Scores")
        plt.savefig(results_dir + save_dir + weights_path[:-4] + "/metrics/mutual_info_scores/scores_hist.jpg")

        # save histograms for each class
        for i, gene in enumerate(classes):
            plt.figure(figsize=(12, 6))
            sns.histplot(MI_scores_per_class[gene])
            plt.xlabel('Mutual Information (MI)')
            plt.title('Distribution of Mutual Information Scores for Gene {}'.format(gene))
            plt.savefig(results_dir + save_dir + weights_path[:-4] + "/metrics/mutual_info_scores/" + "MI_hist_{}.jpg".format(gene))

            if save_most_similar:
                save_as = results_dir + save_dir + weights_path[:-4] + "/metrics/mutual_info_scores/" + \
                          "{}_5most_similar.jpg".format(gene)
                save_image(top5_similar_images_per_class[gene], save_as)

            if save_most_different:
                save_as = results_dir + save_dir + weights_path[:-4] + "/metrics/mutual_info_scores/" +\
                          "{}_5most_different.jpg".format(gene)
                save_image(top5_different_images_per_class[gene], save_as)

    else:
        pass

# save FID scores
if calc_fid_imagenet:
    # make directory for storing metrics
    os.makedirs(results_dir + save_dir + weights_path[:-4] + "/metrics/fid_imagenet", exist_ok=True)

    if classes is not None:
        # convert dict into a pandas dataframe
        fid_df_v1 = pd.DataFrame.from_dict(fid_imagenet_per_class, orient='index')
        fid_df_v1.to_csv(results_dir+save_dir+weights_path[:-4]+"/metrics/fid_imagenet/fid_imagenet_scores_per_class.csv")
        # save table as latex - useful for reports
        with open(results_dir+save_dir+weights_path[:-4]+"/metrics/fid_imagenet/fid_imagenet_scores_latex.txt", 'w') as f:
            f.write(fid_df_v1.to_latex())
    else:
        pass

if calc_fid_eye2gene:
    # make directory for storing metrics
    os.makedirs(results_dir + save_dir + weights_path[:-4] + "/metrics/fid_eye2gene", exist_ok=True)

    if classes is not None:
        # convert dict into a pandas dataframe
        fid_df_v1 = pd.DataFrame.from_dict(fid_eye2gene_per_class, orient='index')
        fid_df_v1.to_csv(results_dir+save_dir+weights_path[:-4]+"/metrics/fid_eye2gene/fid_eye2gene_scores_per_class.csv")
        # save table as latex - useful for reports
        with open(results_dir+save_dir+weights_path[:-4]+"/metrics/fid_eye2gene/fid_eye2gene_scores_latex.txt", 'w') as f:
            f.write(fid_df_v1.to_latex())
    else:
        pass

if class_preds_eye2gene:
    plt.figure(figsize=(20, 10))
    sns.heatmap(confusions, annot=True)
    plt.xlabel("Actual Class")
    plt.ylabel("Predicted Class")
    plt.show()
    plt.savefig(results_dir+save_dir+weights_path[:-4]+"/metrics/confusion_matrix.jpg")

