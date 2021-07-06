"""
This module contains all the helper functions for dealing with image data and model
"""

# import libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from PIL import Image
import glob
import matplotlib.pyplot as plt


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
    def __init__(self, images_directory, image_labels=None, transforms=None, labels_transform=None):
        self.img_dir = glob.glob(images_directory+"*.jpeg")
        if len(self.img_dir) == 0:
            self.img_dir = glob.glob(images_directory+"*.jpg")
        if len(self.img_dir) == 0:
            self.img_dir = glob.glob(images_directory+"*.png")
        self.img_labels = image_labels
        self.transform = transforms
        self.lbl_transform = labels_transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, item):
        image = Image.open(self.img_dir[item])
        # TODO: need to change this for conditional generation model!
        label = 0 if self.img_labels is None else self.img_labels[item]
        if self.transform:
            image = self.transform(image)
        if self.lbl_transform:
            label = self.lbl_transform(label)
        return image, label


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
