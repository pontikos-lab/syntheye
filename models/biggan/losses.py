import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real, **kwargs):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1 + L2


def loss_dcgan_gen(dis_fake, *args):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, **kwargs):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real + loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake, *args):
  loss = -torch.mean(dis_fake)
  return loss


# WGAN-GP loss
def wgan_disc_loss(dis_fake, dis_real, **kwargs):
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
  epsilon = torch.rand(len(kwargs['reals']), 1, 1, 1, device=kwargs['device'], requires_grad=True)
  from helpers.data_utils import get_gradient_biggan, gradient_penalty
  gradient = get_gradient_biggan(kwargs['D'], kwargs['reals'], kwargs['dy'], kwargs['fakes'].detach(), kwargs['gy'].detach(), epsilon)
  c_lambda = 10
  gp = gradient_penalty(gradient)
  disc_loss = torch.mean(dis_fake) - torch.mean(dis_real) + c_lambda * gp
  return disc_loss


def wgan_gen_loss(dis_fake, *args):
  """
  Return the loss of a generator given the critic's scores of the generator's fake images.
  Parameters:
      disc_fake_pred: the critic's scores of the fake images
  Returns:
      gen_loss: a scalar loss value for the current batch of the generator
  """
  gen_loss = -1 * torch.mean(dis_fake)
  return gen_loss


# relativistic GAN loss
def loss_RAHinge_dis(dis_fake, dis_real, **kwargs):

  # difference between real and fake:
  r_f_diff = dis_real - torch.mean(dis_fake)

  # difference between fake and real samples
  f_r_diff = dis_fake - torch.mean(dis_real)

  # return the loss
  loss_real = torch.mean(torch.nn.ReLU()(1 - r_f_diff))
  loss_fake = torch.mean(torch.nn.ReLU()(1 + f_r_diff))
  return loss_real + loss_fake


def loss_RAHinge_gen(dis_fake, dis_real, **kwargs):

  # difference between real and fake:
  r_f_diff = dis_real - torch.mean(dis_fake)

  # difference between fake and real samples
  f_r_diff = dis_fake - torch.mean(dis_real)

  # return the loss
  return torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))

# Default to hinge loss
# generator_loss = wgan_gen_loss
# discriminator_loss = wgan_disc_loss
