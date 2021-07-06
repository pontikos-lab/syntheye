"""
Module for WGAN with gradient penalty architecture -> https://arxiv.org/abs/1704.00028
Code taken from Coursera "Building Basic Generative Adversarial Networks (GANs)" Week 3
"""

from torch import nn

# set dimensions of the wgan
nc = 1 # number of channels in output
ngf = 64 # dimension of generator feature maps
ndf = 64 # dimension of discriminator feature maps


# MODEL ARCHITECTURE CLASSES
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
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
                nn.Tanh(),
            )

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


class Critic(nn.Module):
    """
    Critic Class
    Values:
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    """
    def __init__(self, im_chan=nc, hidden_dim=ndf):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_crit_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_crit_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_crit_block(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            self.make_crit_block(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, bias=True, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding, bias=bias),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding, bias=bias),
            )

    def forward(self, image):
        """
        Function for completing a forward pass of the critic: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        """
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)
