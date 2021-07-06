"""
Module for Progressive Growing GANs Architecture -> https://arxiv.org/pdf/1710.10196.pdf
Code modified from https://github.com/rosinality/progressive-gan-pytorch
"""

import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2/fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name="weight"):
    EqualLR.apply(module, name)
    return module


class PixelNorm(nn.Module):
    """ Performs pixel-wise normalization of feature maps """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualConv2d(nn.Module):
    """ Equalized weights and bias for Conv2D """
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        # initialize weights to normal
        conv.weight.data.normal_()
        # set bias=0
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualConvTranspose2d(nn.Module):
    """ Equalized weights and bias for transposed Conv2D """
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    """ Equalized weights and bias for linear operation """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input.squeeze())


class MiniBatch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        out_std = torch.sqrt(input.var(0, unbiased=False) + 1e-8)
        mean_std = out_std.mean().expand(input.size(0), 1, 4, 4)
        out = torch.cat([input, mean_std], dim=1)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=512, in_chan=512, mode="grayscale"):
        super().__init__()
        # tracks how large the spatial resolution gets
        self.res_depth = 0
        # alpha - controls how much to weight the feature maps
        self.alpha = 0
        # latent input
        self.z_dim = z_dim
        # number of input channels in spatial resolution
        self.in_chan = in_chan
        # mode of image
        self.mode = mode

        self.input_layer = nn.Sequential(EqualConvTranspose2d(z_dim, in_chan, 4, 1, 0),
                                        PixelNorm(),
                                        nn.LeakyReLU(0.2),
                                        EqualConv2d(in_chan, in_chan, 3, 1, 1),
                                        PixelNorm(),
                                        nn.LeakyReLU(0.2))

        # convolution blocks
        self.progression_blocks = nn.ModuleList([self.input_layer,
                                                 self.add_upsampling_block(in_chan, in_chan),
                                                 self.add_upsampling_block(in_chan, in_chan),
                                                 self.add_upsampling_block(in_chan, in_chan),
                                                 self.add_upsampling_block(in_chan, in_chan//2),
                                                 self.add_upsampling_block(in_chan//2, in_chan//4),
                                                self.add_upsampling_block(in_chan//4, in_chan//8),
                                                self.add_upsampling_block(in_chan//8, in_chan//16),
                                                 self.add_upsampling_block(in_chan//16, in_chan//32)])

        self.toRGBblocks = nn.ModuleList([self.toRGB(in_chan),
                                          self.toRGB(in_chan),
                                          self.toRGB(in_chan),
                                          self.toRGB(in_chan),
                                          self.toRGB(in_chan//2),
                                          self.toRGB(in_chan//4),
                                          self.toRGB(in_chan//8),
                                          self.toRGB(in_chan//16),
                                          self.toRGB(in_chan//32)])

    def toRGB(self, in_chan):
        if self.mode == "grayscale":
            return EqualConv2d(in_chan, 1, 1, 1)
        else:
            return EqualConv2d(in_chan, 3, 1, 1)

    def add_upsampling_block(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, final_block=False):
        if not final_block:
            conv_block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                       EqualConv2d(in_chan, out_chan, kernel_size, stride, padding),
                                       PixelNorm(),
                                       nn.LeakyReLU(0.2),
                                       EqualConv2d(out_chan, out_chan, kernel_size, stride, padding),
                                       PixelNorm(),
                                       nn.LeakyReLU(0.2))
        else:
            conv_block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                       EqualConv2d(in_chan, out_chan, kernel_size, stride, padding),
                                       PixelNorm(),
                                       nn.LeakyReLU(0.2),
                                       EqualConv2d(out_chan, out_chan, kernel_size, stride, padding),
                                       PixelNorm())
        return conv_block

    def combine_blocks(self, out1, out2, torgb1, torgb2):
        # convert out1 and out2 to rgb
        out1 = torgb1(nn.Upsample(scale_factor=2, mode="nearest")(out1))
        out2 = torgb2(out2)
        return nn.Tanh()(self.alpha*out2 + (1-self.alpha)*out1)

    def forward(self, noise):
        # reshape latent noise vector into (-1, z_dim, 1, 1)
        x = noise.view(len(noise), self.z_dim, 1, 1)
        # upscale to 4x4 resolution
        x = self.progression_blocks[0](x)
        if self.res_depth == 0:
            return nn.Tanh()(self.toRGBblocks[0](x))
        # phase in new blocks - depends on the value of self.res_depth
        out1 = x
        out2 = x
        for i in range(1, self.res_depth+1):
            out1 = self.progression_blocks[i-1](out1) if i-1 > 0 else out1
            out2 = self.progression_blocks[i](out2)
            # print(out1.shape, out2.shape)
            if i == self.res_depth:
                return self.combine_blocks(out1, out2, self.toRGBblocks[i-1], self.toRGBblocks[i])


class Discriminator(nn.Module):
    def __init__(self, feat_chan=512, mode="grayscale"):
        super().__init__()
        # tracks how large the spatial resolution gets
        self.res_depth = 0
        # alpha - controls how much weight the feature maps
        self.alpha = 0
        # mode of image
        self.mode = mode
        # fromRGB blocks
        self.fromRGBblocks = nn.ModuleList([self.fromRGB(feat_chan),
                                            self.fromRGB(feat_chan),
                                            self.fromRGB(feat_chan),
                                            self.fromRGB(feat_chan),
                                            self.fromRGB(feat_chan//2),
                                            self.fromRGB(feat_chan//4),
                                            self.fromRGB(feat_chan//8),
                                            self.fromRGB(feat_chan//16),
                                            self.fromRGB(feat_chan//32)])

        # downsampling blocks
        self.downsampling_blocks = nn.ModuleList([self.add_downsampling_block(feat_chan, feat_chan, final_block=True),
                                                  self.add_downsampling_block(feat_chan, feat_chan),
                                                  self.add_downsampling_block(feat_chan, feat_chan),
                                                  self.add_downsampling_block(feat_chan, feat_chan),
                                                  self.add_downsampling_block(feat_chan//2, feat_chan),
                                                  self.add_downsampling_block(feat_chan//4, feat_chan//2),
                                                  self.add_downsampling_block(feat_chan//8, feat_chan//4),
                                                  self.add_downsampling_block(feat_chan//16, feat_chan//8),
                                                  self.add_downsampling_block(feat_chan//32, feat_chan//16)])

    def fromRGB(self, out_chan):
        if self.mode == "grayscale":
            return EqualConv2d(1, out_chan, 1, 1)
        else:
            return EqualConv2d(3, out_chan, 1, 1)

    def combine_blocks(self, out1, out2):
        return self.alpha*out1 + (1 - self.alpha)*out2

    def add_downsampling_block(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, final_block=False):
        if not final_block:
            conv_block = nn.Sequential(EqualConv2d(in_chan, out_chan, kernel_size, stride, padding),
                                       PixelNorm(),
                                       nn.LeakyReLU(0.2),
                                       EqualConv2d(out_chan, out_chan, kernel_size, stride, padding),
                                       PixelNorm(),
                                       nn.LeakyReLU(0.2),
                                       nn.AvgPool2d(kernel_size=2))
        else:
            conv_block = nn.Sequential(MiniBatch(), # minibatch layer adds an extra channel
                                       EqualConv2d(in_chan+1, out_chan, kernel_size, stride, padding),
                                       PixelNorm(),
                                       nn.LeakyReLU(0.2),
                                       EqualConv2d(out_chan, out_chan, 4, 1, 0),
                                       PixelNorm(),
                                       nn.LeakyReLU(0.2),
                                       EqualLinear(out_chan, 1))
        return conv_block

    def forward(self, input):
        # increase number of channels on RGB input
        x = self.fromRGBblocks[self.res_depth](input)
        x = nn.LeakyReLU(0.2)(x)

        # perform down-sampling
        for i in range(self.res_depth, 0, -1):
            x = self.downsampling_blocks[i](x)
            if i == self.res_depth:
                downscaled_input = F.avg_pool2d(input, 2)
                x = self.combine_blocks(self.fromRGBblocks[i-1](downscaled_input), x)

        # last layer - includes mini-batch layer
        x = self.downsampling_blocks[0](x)
        return x
