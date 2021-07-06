""" Loss functions for GANs """

import torch
import torch.nn as nn


def dcgan_disc_loss(disc_fake_pred, disc_real_pred, **kwargs):
    """
    Return the loss of a critic given the critic's scores for fake and real images.
    Parameters:
        disc_fake_pred: the critic's scores of the fake images
        disc_real_pred: the critic's scores of the real images
    Returns:
        disc_loss: a scalar for the critic's loss, accounting for the relevant factors
    """
    disc_fake_loss = nn.BCEWithLogitsLoss()(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_loss = nn.BCEWithLogitsLoss()(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


def wgan_gen_loss(disc_fake_pred, *args):
    """
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        disc_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    """
    gen_loss = -1 * torch.mean(disc_fake_pred)
    return gen_loss


def wgan_disc_loss(disc_fake_pred, disc_real_pred, **kwargs):
    """
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        disc_fake_pred: the critic's scores of the fake images
        disc_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty
    Returns:
        disc_loss: a scalar for the critic's loss, accounting for the relevant factors
    """
    c_lambda = kwargs['c_lambda']
    gp = kwargs['gp']
    disc_loss = torch.mean(disc_fake_pred) - torch.mean(disc_real_pred) + c_lambda*gp
    return disc_loss
