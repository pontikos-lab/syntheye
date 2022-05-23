""" Module implementing various loss functions
Code taken from -> https://github.com/akanimax/BMSG-GAN
"""

import torch
from torch.nn import functional as F

# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels=None):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels=None):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================
# criterion = nn.L1Loss()
def r1loss(inputs, label=None):
    # non-saturating loss with R1 regularization
    l = -1 if label else 1
    return F.softplus(l*inputs).mean()

class NSGANLoss(GANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels=None):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device
        real_samps.requires_grad = True

        # predictions for real images and fake images separately :
        real_preds = self.dis(real_samps)
        real_loss = F.softplus(-1*real_preds).mean()

        from torch.autograd import grad
        grad_real = grad(outputs=real_preds.sum(), inputs=real_samps, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty = 0.5*10*grad_penalty
        real_loss += grad_penalty

        # calculate the fake loss
        fake_preds = self.dis(fake_samps)
        fake_loss = F.softplus(fake_preds).mean()

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, labels=None):
        preds = self.dis(fake_samps)
        loss = F.softplus(-1*preds).mean()
        return loss

class BCEwithCE(GANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss
        from torch.nn import CrossEntropyLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.bce = BCEWithLogitsLoss()
        self.ce = CrossEntropyLoss()

    def dis_loss(self, real_samps, fake_samps, labels=None):

        # predictions for real images and fake images separately :
        r_preds, r_class_preds = self.dis(real_samps)
        f_preds, f_class_preds = self.dis(fake_samps)

        # calculate the real loss: r_preds shape = [n_samples]
        real_bce_loss = self.bce(r_preds, torch.ones(r_preds.shape[0]).to(r_preds.device))
        fake_bce_loss = self.bce(f_preds, torch.zeros(f_preds.shape[0]).to(f_preds.device))
        total_bce_loss = (real_bce_loss + fake_bce_loss) / 2

        # calculate cross entropy loss on the class predictions
        real_ce_loss = self.ce(r_class_preds, labels)
        fake_ce_loss = self.ce(f_class_preds, labels)
        total_ce_loss = (real_ce_loss + fake_ce_loss) / 2

        return total_bce_loss + total_ce_loss, total_ce_loss

    def gen_loss(self, _, fake_samps, labels=None):
        f_preds, f_class_preds = self.dis(fake_samps)
        bce_loss = self.bce(f_preds, torch.ones(f_preds.shape[0]).to(f_preds.device))
        ce_loss = self.ce(f_class_preds, labels)
        return bce_loss + ce_loss


class WGAN_GP(GANLoss):

    def __init__(self, dis, drift=0.001, use_gp=False):
        super().__init__(dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, labels=None, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """

        def _interp(real_sample, fake_sample, epsilon):
            # create an interpolation of real and fake samples
            merged = (epsilon * real_sample) + ((1 - epsilon) * fake_sample)
            merged.requires_grad = True
            return merged

        # generate random epsilon
        batch_size = real_samps[0].shape[0]
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samps[0].device)

        # create the merge of both real and fake samples
        if isinstance(real_samps, list):
            merged = [_interp(real_samps[i], fake_samps[i], epsilon) for i in range(len(real_samps))]
        else:
            merged = _interp(real_samps, fake_samps, epsilon)

        # forward pass
        op = self.dis(merged, labels)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=torch.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, labels=None):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps, labels)
        real_out = self.dis(real_samps, labels)

        loss = (torch.mean(fake_out) - torch.mean(real_out)
                + (self.drift * torch.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps, labels)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps, labels=None):
        # calculate the WGAN loss for generator
        loss = -torch.mean(self.dis(fake_samps, labels))

        return loss


class LSGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels=None):
        return 0.5 * (((torch.mean(self.dis(real_samps)) - 1) ** 2)
                      + (torch.mean(self.dis(fake_samps))) ** 2)

    def gen_loss(self, _, fake_samps, labels=None):
        return 0.5 * ((torch.mean(self.dis(fake_samps)) - 1) ** 2)


class LSGAN_SIGMOID(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels=None):
        from torch.nn.functional import sigmoid
        real_scores = torch.mean(sigmoid(self.dis(real_samps)))
        fake_scores = torch.mean(sigmoid(self.dis(fake_samps)))
        return 0.5 * (((real_scores - 1) ** 2) + (fake_scores ** 2))

    def gen_loss(self, _, fake_samps, labels=None):
        from torch.nn.functional import sigmoid
        scores = torch.mean(sigmoid(self.dis(fake_samps)))
        return 0.5 * ((scores - 1) ** 2)


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels=None):
        r_preds, r_mus, r_sigmas = self.dis(real_samps)
        f_preds, f_mus, f_sigmas = self.dis(fake_samps)

        loss = (torch.mean(torch.nn.ReLU()(1 - r_preds)) +
                torch.mean(torch.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, labels=None):
        return -torch.mean(self.dis(fake_samps))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels=None):
        # Obtain predictions
        r_preds = self.dis(real_samps, labels)
        f_preds = self.dis(fake_samps, labels)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, labels=None):
        # Obtain predictions
        r_preds = self.dis(real_samps, labels)
        f_preds = self.dis(fake_samps, labels)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))


class RAHingeWithCrossEntropy(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, labels=None):

        # Obtain predictions
        r_preds, r_pred_classes = self.dis(real_samps)
        f_preds, f_pred_classes = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the RAHinge loss
        hinge_loss = torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff))

        from torch.nn import CrossEntropyLoss
        # return the cross entropy loss for classes
        ce_loss = CrossEntropyLoss()(r_pred_classes, labels) + CrossEntropyLoss()(f_pred_classes, labels)

        return hinge_loss + ce_loss, ce_loss

    def gen_loss(self, real_samps, fake_samps, labels=None):
        # Obtain predictions
        r_preds, r_pred_classes = self.dis(real_samps, labels)
        f_preds, f_pred_classes = self.dis(fake_samps, labels)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        from torch.nn import CrossEntropyLoss
        # return the cross entropy loss for classes
        ce_loss = CrossEntropyLoss()(r_pred_classes, labels) + CrossEntropyLoss()(f_pred_classes, labels)

        # return the loss
        return torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff)) + ce_loss