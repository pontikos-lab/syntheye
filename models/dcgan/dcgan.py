"""
Module for DCGAN -> https://arxiv.org/abs/1511.06434
Code taken from Coursera "Building Basic Generative Adversarial Networks (GANs)" Week 2
"""

from torch import nn


# set dimensions of the dcgan
nc = 1 # number of channels in output
ngf = 64 # dimension of generator feature maps
ndf = 64 # dimension of discriminator feature maps


class Generator(nn.Module):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, z_dim=100, im_chan=nc, hidden_dim=ngf):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, final_layer=False),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, padding=1, bias=False, final_layer=True)
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, bias=True, final_layer=False):

        """
        Function to return a sequence of operations corresponding to a generator block of DCGAN,
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            padding: pads the feature map
            bias: includes a bias parameter in model parameters
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(num_features=output_channels),
                nn.ReLU(True)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


class Discriminator(nn.Module):

    """
    Discriminator Class
    Values:
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, im_chan=nc, hidden_dim=ndf):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_disc_block(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, bias=True, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN,
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            padding: pads the feature maps
            bias: includes bias term in model parameters
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding, bias=bias),
                nn.BatchNorm2d(num_features=output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding, bias=bias)
            )

    """
    Function for completing a forward pass of the discriminator: Given an image tensor, 
    returns a 1-dimension tensor representing fake/real.
    Parameters:
        image: a flattened image tensor with dimension (im_dim)
    """

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)