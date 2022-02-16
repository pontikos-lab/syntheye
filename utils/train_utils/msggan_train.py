"""
Trainer script for multi-scale gradients GAN (MSGGAN) -> https://arxiv.org/abs/1710.10196
Code adapted from -> https://github.com/akanimax/BMSG-GAN
"""

# import modules
from utils.data_utils import *
from models.msggan import losses
from models.acgan.losses import RelativisticAverageHingeGAN


def train(model, train_dataloader, test_dataloader, train_configs, **kwargs):

    # extract config values
    epochs = train_configs['epochs']
    loss_fn = train_configs['loss_fn']
    gen_lr = train_configs['gen_lr']
    disc_lr = train_configs['disc_lr']
    beta1 = train_configs['beta1']
    beta2 = train_configs['beta2']
    checkpoint_factor = train_configs['save_checkpoint_steps']
    n_samples = train_configs['n_samples_to_generate']
    n_disc_updates = train_configs['n_disc_updates']
    display_step = train_configs['display_step']
    checkpoint_path = train_configs['train_from_checkpoint']
    ema_path = train_configs['train_from_ema']

    # set loss function
    if loss_fn == "Hinge":
        loss = losses.HingeGAN
    elif loss_fn == "RAHinge":
        loss = losses.RelativisticAverageHingeGAN
    elif loss_fn == "RAHingeCE":
        loss = losses.RAHingeWithCrossEntropy
    elif loss_fn == "BCEwithCE":
        loss = losses.BCEwithCE
    elif loss_fn == "standard-gan":
        loss = losses.StandardGAN
    elif loss_fn == "nsgan":
        loss = losses.NSGANLoss
    elif loss_fn == "lsgan":
        loss = losses.LSGAN
    elif loss_fn == "lsgan-sigmoid":
        loss = losses.LSGAN_SIGMOID
    elif loss_fn == "wgan-gp":
        def loss(dis):
            return losses.WGAN_GP(dis, use_gp=True)
    else:
        raise Exception("Unknown loss function requested")

    # set optimizers
    if loss_fn == "wgan-gp":
        gen_optim = torch.optim.RMSprop(model.gen.parameters(), lr=gen_lr)
        disc_optim = torch.optim.RMSprop(model.dis.parameters(), lr=disc_lr)
    else:
        gen_optim = torch.optim.Adam(model.gen.parameters(), lr=gen_lr, betas=(beta1, beta2))
        disc_optim = torch.optim.Adam(model.dis.parameters(), lr=disc_lr, betas=(beta1, beta2))

    if checkpoint_path is not None:
        model_state_dict = torch.load(checkpoint_path)
        start = model_state_dict["epoch"]
        global_step = model_state_dict["i"]
        model.gen.load_state_dict(model_state_dict["gen_state_dict"])
        model.dis.load_state_dict(model_state_dict["disc_state_dict"])
        model.gen_shadow.load_state_dict(torch.load(ema_path))
        gen_optim.load_state_dict(model_state_dict["gen_optim_state_dict"])
        disc_optim.load_state_dict(model_state_dict["disc_optim_state_dict"])
    else:
        start = 1

    # train the gan
    model.train(train_dataloader, test_dataloader, gen_optim, disc_optim, n_disc_updates=n_disc_updates,
                loss_fn=loss(model.dis), num_epochs=epochs, checkpoint_factor=checkpoint_factor, num_samples=n_samples,
                display_step=display_step, save_dir=os.path.join("checkpoints/", kwargs["checkpoints_fname"]),
                start=start, log_dir=train_configs['logfile'])

    return model