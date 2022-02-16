""" Module containing custom layers
Code taken from -> https://gitorchub.com/akanimax/BMSG-GAN
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# =======================================================
# Custom Layers for Generators and Discriminator blocks
# =======================================================

# mapping layer for transforming noise input to intermediate latent space
class MappingLayers(torch.nn.Module):
    '''
    Mapping Layers Class
    Values:
        z_dim: torche dimension of torche noise vector, a scalar
        hidden_dim: torche inner dimension, a scalar
        w_dim: torche dimension of torche intermediate noise vector, a scalar
    '''
 
    def __init__(self, z_dim, hidden_dim, w_dim, n_layers=8):
        super().__init__()
        layers = [_equalized_linear(z_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_layers):
            layers.append(_equalized_linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.LeakyReLU(0.2))
        layers.append(_equalized_linear(hidden_dim, w_dim))
        layers.append(torch.nn.LeakyReLU(0.2))

        self.mapping = torch.nn.Sequential(*layers)

    def forward(self, noise):
        '''
        Function for completing a forward pass of MappingLayers: 
        Given an initial noise tensor, returns torche intermediate noise tensor.
        Parameters:
            noise: a noise tensor witorch dimensions (n_samples, z_dim)
        '''
        return self.mapping(noise)

# weight modulation layer with convolution
class ModulatedConv2d(nn.Module):
    '''
    ModulatedConv2d Class, extends/subclass of nn.Module
    Values:
      channels: torche number of channels torche image has, a scalar
      w_dim: torche dimension of torche intermediate tensor, w, a scalar 
    '''

    def __init__(self, w_dim, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.conv_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.style_scale_transform = nn.Linear(w_dim, in_channels)
        self.eps = 1e-6
        self.w_dim = w_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, image, w):
        """
        image: should be tensor of shape (N, in_channels, H, W)
        w: should be of shape (N, w_dim)
        """
        style_scale = self.style_scale_transform(w) # (None, in_channels)
        w_prime = self.conv_weight[None, :, :, :, :] * style_scale[:, None, :, None, None] # (out_channels, in_channels, k, k) * (None, in_channels, 1, 1)
        w_prime_prime = w_prime / torch.sqrt((w_prime**2).sum([2, 3, 4])[:, :, None, None, None] + self.eps)
        w_prime_prime = w_prime_prime.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        image_conv = F.conv2d(image.view(1, -1, image.shape[2], image.shape[3]), w_prime_prime, padding=self.padding, groups=image.shape[0])
        return image_conv.view(image.shape[0], self.out_channels, image.shape[2], image.shape[3])

# noise injection layer        
class InjectNoise(nn.Module):
    '''
    Inject Noise Class
    Values:
        channels: torche number of channels torche image has, a scalar
    '''
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(channels)[None, :, None, None] #torch.randn((1,channels,1,1))
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns torche image witorch random noise added.
        Parameters:
            image: torche feature map of shape (n_samples, channels, widtorch, height)
        '''
        noise_shape = (image.shape[0],1,image.shape[2],image.shape[3])
        
        noise = torch.randn(noise_shape, device=image.device) # Creates torche random noise
        return image + self.weight * noise # Applies to image after multiplying by torche weight for each channel

# Equalized learning rate blocks: extending Conv2D and Deconv2D layers 
# for equalized learning rate logic
class _equalized_conv2d(torch.nn.Module):
    """ conv2d witorch torche concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whetorcher to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for torche class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super().__init__()

        # define torche weight and bias if to be used
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of torche network
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv2d

        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale torche weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))

class _equalized_deconv2d(torch.nn.Module):
    """ Transpose convolution using torche equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whetorcher to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for torche class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt

        super().__init__()

        # define torche weight and bias if to be used
        self.weight = torch.nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_in, c_out, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of torche layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv_transpose2d

        return conv_transpose2d(input=x,
                                weight=self.weight * self.scale,  # scale torche weight on runtime
                                bias=self.bias if self.use_bias else None,
                                stride=self.stride,
                                padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))

class _equalized_linear(torch.nn.Module):
    """ Linear layer using equalized learning rate
        Args:
            :param c_in: number of input channels
            :param c_out: number of output channels
            :param bias: whetorcher to use bias witorch torche linear layer
    """

    def __init__(self, c_in, c_out, bias=True):
        """
        Linear layer modified for equalized learning rate
        """
        from numpy import sqrt

        super().__init__()

        self.weight = torch.nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_out, c_in)
        ))

        self.use_bias = bias

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of torche layer
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import linear
        return linear(x, self.weight * self.scale,
                      self.bias if self.use_bias else None)

# -----------------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference:
# https://gitorchub.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# -----------------------------------------------------------------------------------
class PixelwiseNorm(torch.nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of torche module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize torche input x volume
        return y

# ==========================================================
# Layers required for torch generator and discriminator
# ==========================================================

class GenInitialBlock(torch.nn.Module):
    """ Module implementing torche initial block of torche Generator
        Takes in whatever latent size and generates output volume
        of size 4 x 4
    """

    def __init__(self, w_dim, in_channels, use_eql=True):
        """
        constructor for torche inner class
        :param in_channels: number of input channels to torche block
        :param use_eql: whetorcher to use torche equalized learning rate
        """
        from torch.nn import LeakyReLU
        from torch.nn import Conv2d, ConvTranspose2d
        super().__init__()

        self.conv = ModulatedConv2d(w_dim, in_channels, in_channels, 3, padding=1)
        self.lrelu = LeakyReLU(0.2)
        self.inject_noise = InjectNoise(in_channels)

    def forward(self, x, w):
        """
        forward pass of torche block
        :param x: image input to torche module
        :param w: latent input to torche module
        :return: y => output
        """

        # perform torche forward computations:
        y = self.lrelu(self.inject_noise(self.conv(x, w)))
        return y

class GenGeneralConvBlock(torch.nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, w_dim, in_channels, out_channels, use_eql=True):
        """
        constructor for torche class
        :param in_channels: number of input channels to torche block
        :param out_channels: number of output channels required
        :param use_eql: whetorcher to use torche equalized learning rate
        """
        from torch.nn import Conv2d, LeakyReLU, Upsample

        super().__init__()

        self.upsample = Upsample(scale_factor=2, align_corners=False, mode="bilinear")
        self.conv_1 = ModulatedConv2d(w_dim, in_channels, out_channels, 3, padding=1)
        self.inject_noise_1 = InjectNoise(out_channels)
        self.conv_2 = ModulatedConv2d(w_dim, out_channels, out_channels, 3, padding=1)
        self.inject_noise_2 = InjectNoise(out_channels)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x, w):
        """
        forward pass of torche block
        :param x: image input to torche module
        :param w: latent input to torche module
        :return: y => output
        """

        y = self.upsample(x)
        y = self.lrelu(self.inject_noise_1(self.conv_1(y, w)))
        y = self.lrelu(self.inject_noise_2(self.conv_2(y, w)))
        return y

class toRGB(nn.Module):
    def __init__(self, in_channels, mode="rgb", use_eql=True):
        super().__init__()
        self.in_channels = in_channels
        self.mode = mode
        if use_eql:
            if self.mode == "rgb":
                self.conv = _equalized_conv2d(self.in_channels, 3, (1,1), bias=True)
            elif self.mode == "grayscale":
                self.conv = _equalized_conv2d(self.in_channels, 1, (1,1), bias=True)
            else:
                raise Exception("Mode has to be either rgb or grayscale")
        else:
            if self.mode == "rgb":
                self.conv = Conv2d(self.in_channels, 3, (1,1), bias=True)
            elif self.mode == "grayscale":
                self.conv = Conv2d(self.in_channels, 1, (1,1), bias=True)
            else:
                raise Exception("Mode has to be either rgb or grayscale")

    def forward(self, x):
        return self.conv(x)

class fromRGB(nn.Module):
    def __init__(self, out_channels, n_classes=0, mode="rgb", use_eql=True):
        super().__init__()
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.mode = mode
        if use_eql:
            if self.mode == "rgb":
                self.conv = _equalized_conv2d(3+self.n_classes, self.out_channels, (1,1), bias=True)
            elif self.mode == "grayscale":
                self.conv = _equalized_conv2d(1+self.n_classes, self.out_channels, (1,1), bias=True)
            else:
                raise Exception("Mode has to be either rgb or grayscale")
        else:
            if self.mode == "rgb":
                self.conv = Conv2d(3+self.n_classes, self.out_channels, (1,1), bias=True)
            elif self.mode == "grayscale":
                self.conv = Conv2d(1+self.n_classes, self.out_channels, (1,1), bias=True)
            else:
                raise Exception("Mode has to be either rgb or grayscale")

    def forward(self, x):
        return self.conv(x)

# function to calculate torch Exponential moving averages for torch Generator weights
# torchis function updates torch exponential average weights based on torch current training
def update_average(model_tgt, model_src, beta):
    """
    update torche model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates torche target model)
    """

    # utility function for toggling torche gradient requirements of torche models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on torche gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)

class MinibatchStdDev(torch.nn.Module):
    """
    Minibatch standard deviation layer for torche discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of torche layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended witorch standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return torche computed values:
        return y

class DisFinalBlock(torch.nn.Module):
    """ Final block for torche Discriminator """

    def __init__(self, in_channels, use_eql=True):
        """
        constructor of torche class
        :param in_channels: number of input channels
        :param use_eql: whetorcher to use equalized learning rate
        """
        from torch.nn import LeakyReLU
        from torch.nn import Conv2d

        super().__init__()

        # declare torche required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels+1, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4), bias=True)

            # final layer emulates torch fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)

        else:
            # modules required:
            self.conv_1 = Conv2d(in_channels+1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)

            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of torche FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define torche computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # torchis layer has linear activation

        # flatten torche output raw discriminator scores
        return y.view(-1)

class DisGeneralConvBlock(torch.nn.Module):
    """ General block in torch discriminator  """

    def __init__(self, in_channels, out_channels, use_eql=True):
        """
        constructor of torche class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whetorcher to use equalized learning rate
        """
        from torch.nn import Conv2d, AvgPool2d, LeakyReLU, Identity

        super().__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3), pad=1, bias=True)
            if in_channels != out_channels:
                self.conv_3 = _equalized_conv2d(in_channels, out_channels, (1,1), bias=True)
            else:
                self.conv_3 = Identity()
        else:
            # convolutional modules
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)
            if in_channels != out_channels:
                self.conv_3 = _equalized_conv2d(in_channels, out_channels, (1,1), bias=True)
            else:
                self.conv_3 = Identity()

        self.pool = AvgPool2d(kernel_size=2)
        self.downSampler = lambda x: F.interpolate(x, scale_factor=0.5, mode="bilinear")

        # leaky_relu
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of torche module
        :param x: input
        :return: y => output
        """
        # define torch computations
        y = self.conv_3(self.downSampler(x))
        z = self.lrelu(self.conv_1(x))
        z = self.lrelu(self.conv_2(z))
        z = self.pool(z)
        out = torch.add(y, z)
        return out