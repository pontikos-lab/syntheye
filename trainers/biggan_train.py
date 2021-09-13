"""
Modified version of the BigGAN training function.

BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).
    Let's go.
"""

import functools
import os
from tqdm import tqdm
import torch
import models.biggan.BigGAN
from models.biggan import utils, inception_utils, train_fns
from models.biggan.sync_batchnorm import patch_replication_callback
from torch.utils.tensorboard import SummaryWriter
from helpers.data_utils import show_tensor_images


def train(model, dataloader, train_configs, device, **kwargs):

    # extract config values that we will use in this function
    num_epochs = train_configs['epochs']
    checkpoint_factor = train_configs['save_checkpoint_steps']
    n_samples = train_configs['n_samples_to_generate']
    display_step = train_configs['display_step']
    save_dir = "checkpoints/"+kwargs["checkpoints_fname"]
    train_from_checkpoint = train_configs['train_from_checkpoint']

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # ===========================
    # Setup GAN Model
    # ===========================

    if train_configs['ema']:
        G, D, G_ema = model
        ema = utils.ema(G, G_ema, 0.999, 0)
    else:
        G, D = model
        ema, G_ema = None, None

    GD = models.biggan.BigGAN.G_D(G, D)

    # If parallel, parallelize the GD module
    if train_configs['parallel']:
        if train_configs['device_ids'] == 'all':
            device_ids = list(range(torch.cuda.device_count()))
        else:
            device_ids = train_configs['device_ids']
        GD = torch.nn.DataParallel(GD, device_ids=device_ids)
        if train_configs['cross_replica']:
            patch_replication_callback(GD)

    # =====================================
    # CREATE NOISE AND LABEL ARRAY SAMPLERS
    # =====================================

    # create a random sampler
    z_, y_ = utils.prepare_z_y(train_configs['batch_size'], G.dim_z,
                               train_configs['n_classes'], device=device, fp16=False)

    # Prepare another fixed sample z & y to visualize it's evolution throughout training
    fixed_z, fixed_y = utils.prepare_z_y(n_samples, G.dim_z, train_configs['n_classes'], device=device, fp16=False)
    fixed_z.sample_()
    fixed_y.sample_()

    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': train_configs}

    # using trainer from the repo
    train_fn = train_fns.GAN_training_function(G, D, GD, z_, y_, ema, state_dict, train_configs)

    # ===========================
    # BEGIN TRAINING
    # ===========================

    if train_from_checkpoint is not None:
        model_state_dict = torch.load(train_from_checkpoint)
        state_dict['epoch'] = model_state_dict['epoch']
        G.load_state_dict(model_state_dict["gen_state_dict"])
        D.load_state_dict(model_state_dict["disc_state_dict"])
        G.optim.load_state_dict(model_state_dict["gen_optim_state_dict"])
        D.optim.load_state_dict(model_state_dict["disc_optim_state_dict"])
        if train_configs['ema']:
            G_ema.load_state_dict(torch.load(train_configs['train_from_ema']))

    # for visualizing metrics and logs on tensorboard
    writer = SummaryWriter(log_dir=train_configs['logfile'])
    global_step = 0
    for epoch in range(state_dict['epoch'], num_epochs):

        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        # stores losses per epoch
        mean_generator_loss = 0
        mean_discriminator_loss = 0

        for i, (x, y) in enumerate(tqdm(dataloader)):

            # increment iteration counter
            state_dict['itr'] += 1

            # Make sure G and D are in training mode...
            G.train()
            D.train()
            if train_configs['ema']:
                G_ema.train()

            # run the training function on sample
            metrics = train_fn(x.to(device), y.to(device))

            # visualize images every couple of iterations
            if global_step % display_step == 0:
                with torch.no_grad():
                    # create fake images
                    fakes = G_ema(fixed_z, G_ema.shared(fixed_y))
                fake_grid = show_tensor_images(fakes, normalize=True, n_rows=n_samples)
                real_grid = show_tensor_images(x[:n_samples], normalize=True, n_rows=n_samples)
                fake_genes = [dataloader.dataset.idx2class[idx.item()] for idx in fixed_y]
                # real_genes = [dataloader.dataset.idx2class[idx.item()] for idx in y[:n_samples]]
                writer.add_image("Generated classes: {}".format(fake_genes), fake_grid, global_step, dataformats='CHW')
                writer.add_image("Real samples:", real_grid, global_step, dataformats='CHW')

            # compute average gen and disc loss (weighted by batch_size)
            mean_generator_loss += metrics['G_loss'] * len(x)/len(dataloader.dataset)
            # mean_discriminator_loss += ((metrics['D_loss_real'] + metrics['D_loss_fake'])/2)
            # * len(x)/len(dataloader.dataset)
            mean_discriminator_loss += metrics['D_loss'] * len(x)/(2 * len(dataloader.dataset))
            global_step += 1

        # Save weights and copies as configured at specified interval
        if checkpoint_factor != 0:
            if (epoch + 1) % checkpoint_factor == 0:
                os.makedirs(save_dir, exist_ok=True)

                model_state_dict = {"epoch": epoch,
                                    "gen_state_dict": G.state_dict(),
                                    "disc_state_dict": D.state_dict(),
                                    "gen_optim_state_dict": G.optim.state_dict(),
                                    "disc_optim_state_dict": D.optim.state_dict()}

                torch.save(model_state_dict, save_dir + "/model_state_" + str(epoch))

                if train_configs['ema']:
                    gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_"
                                                        + str(epoch) + ".pth")
                    torch.save(G_ema.state_dict(), gen_shadow_save_file)

        # increment epoch counter
        state_dict['epoch'] += 1

        # visualize losses in tensorboard
        writer.add_scalars("Loss", {"G_loss": mean_generator_loss, "D_loss": mean_discriminator_loss}, epoch+1)

    writer.flush()
    writer.close()
