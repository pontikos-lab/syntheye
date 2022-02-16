"""
This module contains all the helper functions for dealing with image data and model
"""

# import libraries
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from PIL import Image
import glob
import matplotlib.pyplot as plt


def get_one_hot_labels(labels, n_classes):
    """
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    """
    return F.one_hot(labels, n_classes)


def combine_vectors(x, y):
    """
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector.
        In this assignment, this will be the noise vector of shape (n_samples, z_dim),
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    """
    combined = torch.cat((x, y), dim=1)
    return combined


def get_noise(n_samples, z_dim, device='cpu'):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """
    return torch.randn(n_samples, z_dim, device=device)


class ImageDataset(Dataset):
    """ PyTorch class for Dataset """
    def __init__(self, data_file, fpath_col_name, lbl_col_name=None, class_vals="all", transforms=None, fold=None, class_mapping=None):
        # read dataframe
        df = pd.read_csv(data_file)
        if lbl_col_name is not None:

            if class_vals == "all":

                if fold is None:
                    self.img_dir = list(df[fpath_col_name])
                    self.img_labels = list(df[lbl_col_name])
                elif fold == "train":
                    train_df = df.where(df.fold.isin([1, 2, 3, 4])).dropna()
                    self.img_dir = list(train_df[fpath_col_name])
                    self.img_labels = list(train_df[lbl_col_name])
                elif fold == "val":
                    val_df = df.where(df.fold == 0).dropna()
                    self.img_dir = list(val_df[fpath_col_name])
                    self.img_labels = list(val_df[lbl_col_name])
                elif fold == "test":
                    test_df = df.where(df.fold == -1).dropna()
                    self.img_dir = list(test_df[fpath_col_name])
                    self.img_labels = list(test_df[lbl_col_name])
                else:
                    raise Exception("fold can be train or test only.")

            else:

                # load selected classes
                if type(class_vals) == str:
                    with open(class_vals, 'r') as f:
                        selected_classes = f.read().splitlines()
                else:
                    selected_classes = class_vals

                # get rows of dataframe for selected classes
                df_subset = df.loc[df[lbl_col_name].isin(selected_classes)]
                if fold is None:
                    self.img_dir = list(df_subset[fpath_col_name])
                    self.img_labels = list(df_subset[lbl_col_name])
                elif fold == "train":
                    train_df = df_subset.where(df_subset.fold.isin([1, 2, 3, 4])).dropna()
                    self.img_dir = list(train_df[fpath_col_name])
                    self.img_labels = list(train_df[lbl_col_name])
                elif fold == "val":
                    val_df = df_subset.where(df_subset.fold == 0).dropna()
                    self.img_dir = list(val_df[fpath_col_name])
                    self.img_labels = list(val_df[lbl_col_name])
                elif fold == "test":
                    test_df = df_subset.where(df_subset.fold == -1).dropna()
                    self.img_dir = list(test_df[fpath_col_name])
                    self.img_labels = list(test_df[lbl_col_name])
                else:
                    raise Exception("fold can be train or test only.")

        else:
            if fold is None:
                self.img_dir = list(df[fpath_col_name])
            elif fold == "train":
                train_df = df.where(df.fold != -1).dropna()
                self.img_dir = list(train_df[fpath_col_name])
            elif fold == "test":
                test_df = df.where(df.fold == -1).dropna()
                self.img_dir = list(test_df[fpath_col_name])
            else:
                raise Exception("fold can be train or test only.")
            self.img_labels = None

        # determine classes and mappings from dataset or from a provided dictionary json file
        if (class_mapping is not None) and (self.img_labels is not None):
            import json
            with open(class_mapping, 'r') as f:
                self.class2idx = json.load(f)
            self.classes = list(self.class2idx.keys())
            self.n_classes = len(self.classes)
            self.idx2class = {v:k for (k,v) in self.class2idx.items()}
        elif self.img_labels is not None:
            # relevant attributes if classes are provided
            self.classes = list(np.unique(self.img_labels))
            self.n_classes = len(self.classes)
            self.idx2class = dict(zip(range(self.n_classes), self.classes))
            self.class2idx = dict(zip(self.classes, range(self.n_classes)))
        else:
            self.classes = None
            self.n_classes = None
            self.idx2class = None
            self.class2idx = None

        # image transformations list
        self.transform = transforms

    def __len__(self):
        return len(self.img_dir)

    def get_samples(self, n_images):
        # stores the real images
        real_images = torch.zeros(n_images, 256, 256)
        filepaths = np.random.choice(len(self.img_dir), size=n_images)
        for i, f in enumerate(filepaths):
            real_images[i, :, :] = self.__getitem__(f)[0]

        return real_images

    def __getitem__(self, item):
        # create PIL object of item-th image
        image = Image.open(self.img_dir[item])
        # get the label index for the item-th image - this is just -1 if no labels are found in datafile
        label = self.class2idx[self.img_labels[item]] if self.img_labels is not None else -1

        # transform images
        if self.transform is not None:
            image = self.transform(image)

        return item, self.img_dir[item], image, label


def weights_init(m):
    """ Initializes model weights to the normal distribution with mean 0 and std 0.02"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def show_tensor_images(image_tensor, n_rows=5, show_image=False, save_path=None, normalize=True):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    # create image grid
    if normalize: # normalize [-1, 1] image to [0, 1]
        image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat, nrow=n_rows)
    # show image grid
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if save_path:
        plt.savefig(save_path)
    if show_image:
        plt.show()
    return image_grid


def get_gradient(disc, real, fake, epsilon):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        disc: the discriminator/critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = disc(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=mixed_images,
        outputs=mixed_scores,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    """
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    """
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gradient_msggan(disc, real, fake, epsilon):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        disc: the discriminator/critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = []
    for i in range(len(real)):
        mixed_images.append(real[i]*epsilon + fake[i]*(1-epsilon))

    # Calculate the critic's scores on the mixed images
    mixed_scores = disc(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=mixed_images,
        outputs=mixed_scores,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def get_gradient_biggan(disc, real, r_class, fake, f_class, epsilon):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        disc: the discriminator/critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_classes = disc.embed(r_class) * epsilon + disc.embed(f_class) * (1 - epsilon)
    # Calculate the critic's scores on the mixed images
    mixed_scores = disc(mixed_images, mixed_classes, y_embedded=True)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=mixed_images,
        outputs=mixed_scores,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient
