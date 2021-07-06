""" TRAINER SCRIPT FOR DCGAN AND WGAN MODELS """

# import libraries
import yaml
from tqdm import tqdm
from helpers.data_utils import *
from torch.utils.tensorboard import SummaryWriter


def load_config(config_file):
    """ Loads model configuration file """
    with open(config_file) as file:
        conf = yaml.safe_load(file)
    return conf


# load configs file
CONFIG_PATH = "model_configs.yaml"
config = load_config(CONFIG_PATH)

# setup other training parameters
z_dim = config['z_dim']
epochs = config['epochs']
display_step = config['display_step']
beta_1 = config['beta1']
beta_2 = config['beta2']
lr = config['lr']
# this is required for the wgan-gp
c_lambda = config['c_lambda']
n_disc_updates = config['n_disc_updates']
verbose = config['verbose']


def train(gen, disc, gen_loss_fn, disc_loss_fn, dataloader, device, *args, **kwargs):

    # set optimizers
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    # records training steps
    cur_step = 0

    # saves generator and discriminator/critic losses
    generator_losses = []
    discriminator_losses = []

    # Writes logs to TensorBoard
    writer = SummaryWriter()

    for epoch in range(epochs):
        if verbose:
            print("Epoch {}/{}".format(epoch + 1, epochs))

        # updates the loss of gen and disc
        mean_generator_loss = 0
        mean_discriminator_loss = 0

        for reals, _ in tqdm(dataloader):
            cur_batch_size = len(reals)
            reals = reals.to(device)

            # average discriminator loss for the batch (averaged over number of disc updates)
            mean_iteration_disc_loss = 0

            # update discriminator/critic
            for _ in range(n_disc_updates):
                disc_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fakes = gen(fake_noise)
                disc_fake_pred = disc(fakes.detach())
                disc_real_pred = disc(reals)

                # compute discriminator/critic losses
                epsilon = torch.rand(len(reals), 1, 1, 1, device=device, requires_grad=True)  # only used by wgan-gp
                gradient = get_gradient(disc, reals, fakes.detach(), epsilon)  # only used by wgan-gp
                gp = gradient_penalty(gradient)  # only used by wgan-gp
                disc_loss = disc_loss_fn(disc_fake_pred, disc_real_pred, gp=gp, c_lambda=c_lambda)

                # keep track of the average discriminator loss in this batch
                mean_iteration_disc_loss += disc_loss.item() / n_disc_updates
                # update gradients
                disc_loss.backward(retain_graph=True)
                # update optimizer
                disc_opt.step()

            # Keep track of average discriminator loss (weighted by batch-size)
            mean_discriminator_loss += mean_iteration_disc_loss * (cur_batch_size / len(dataloader.dataset))
            discriminator_losses.append(mean_discriminator_loss)

            # Update generator
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fakes_2 = gen(fake_noise_2)
            disc_fake_pred = disc(fakes_2)
            gen_loss = gen_loss_fn(disc_fake_pred, torch.ones_like(disc_fake_pred))
            generator_losses.append(gen_loss.item())
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the average generator loss (weighted by batch-size)
            mean_generator_loss += gen_loss.item() * (cur_batch_size / len(dataloader.dataset))
            generator_losses.append(mean_generator_loss)

            # Visualization code: observe real and fake images every 500 steps of training
            if cur_step % display_step == 0 and cur_step > 0:
                real_images = show_tensor_images(reals)
                fake_images = show_tensor_images(fakes)
                writer.add_image("Reals", real_images, cur_step, dataformats='CHW')
                writer.add_image("Fakes", fake_images, cur_step, dataformats='CHW')
            cur_step += 1

        # visualize generator and discriminator loss metrics per epoch
        writer.add_scalars("Loss", {'Generator': mean_generator_loss,
                                    'Discriminator': mean_discriminator_loss}, epoch)

        # save generator and discriminator/critic losses
        # generator_losses.append(mean_generator_loss)
        # discriminator_losses.append(mean_discriminator_loss)

    writer.flush()
    writer.close()

    return gen, disc

# records training steps
# cur_step = 0

# saves generator and discriminator/critic losses
# generator_losses = []
# discriminator_losses = []

# Writes logs to TensorBoard
# writer = SummaryWriter()

# for epoch in range(epochs):
#     if config['verbose']:
#         print("Epoch {}/{}".format(epoch+1, epochs))
#
#     updates the loss of gen and disc
    # mean_generator_loss = 0
    # mean_discriminator_loss = 0
    #
    # for reals, _ in tqdm(dataloader):
    #     cur_batch_size = len(reals)
    #     reals = reals.to(device)
    #
    #     average discriminator loss for the batch (averaged over number of disc updates)
        # mean_iteration_disc_loss = 0
        #
        # update discriminator/critic
        # for _ in range(n_disc_updates):
        #     disc_opt.zero_grad()
        #     fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        #     fakes = gen(fake_noise)
        #     disc_fake_pred = disc(fakes.detach())
        #     disc_real_pred = disc(reals)
        #
        #     compute discriminator/critic losses
            # epsilon = torch.rand(len(reals), 1, 1, 1, device=device, requires_grad=True) # only used by wgan-gp
            # gradient = get_gradient(disc, reals, fakes.detach(), epsilon) # only used by wgan-gp
            # gp = gradient_penalty(gradient) # only used by wgan-gp
            # disc_loss = calc_disc_loss(disc_fake_pred, disc_real_pred, gp=gp, c_lambda=c_lambda)
            #
            # keep track of the average discriminator loss in this batch
            # mean_iteration_disc_loss += disc_loss.item() / n_disc_updates
            # update gradients
            # disc_loss.backward(retain_graph=True)
            # update optimizer
            # disc_opt.step()
        #
        # Keep track of average discriminator loss (weighted by batch-size)
        # mean_discriminator_loss += mean_iteration_disc_loss * (cur_batch_size / len(dataloader.dataset))
        # discriminator_losses.append(mean_discriminator_loss)
        #
        # Update generator
        # gen_opt.zero_grad()
        # fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        # fakes_2 = gen(fake_noise_2)
        # disc_fake_pred = disc(fakes_2)
        # gen_loss = calc_gen_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
        # generator_losses.append(gen_loss.item())
        # gen_loss.backward()
        # gen_opt.step()
        #
        # Keep track of the average generator loss (weighted by batch-size)
        # mean_generator_loss += gen_loss.item() * (cur_batch_size / len(dataloader.dataset))
        # generator_losses.append(mean_generator_loss)
        #
        # Visualization code: observe real and fake images every 500 steps of training
        # if cur_step % display_step == 0 and cur_step > 0:
        #     real_images = show_tensor_images(reals)
        #     fake_images = show_tensor_images(fakes)
        #     writer.add_image("Reals", real_images, cur_step, dataformats='CHW')
        #     writer.add_image("Fakes", fake_images, cur_step, dataformats='CHW')
        # cur_step += 1
    #
    # visualize generator and discriminator loss metrics per epoch
    # writer.add_scalars("Loss", {'Generator': mean_generator_loss,
    #                             'Discriminator': mean_discriminator_loss}, epoch)
    #
    # save generator and discriminator/critic losses
    # generator_losses.append(mean_generator_loss)
    # discriminator_losses.append(mean_discriminator_loss)
#
# writer.flush()
# writer.close()
