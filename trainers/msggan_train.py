"""
Trainer script for multi-scale gradients GAN (MSGGAN) -> https://arxiv.org/abs/1710.10196
Code adapted from -> https://github.com/akanimax/BMSG-GAN
"""

# import modules
from helpers.data_utils import *
from models.msggan import losses


def train(model, data_loader, train_configs, **kwargs):

    # extract config values
    epochs = train_configs['epochs']
    loss_fn = train_configs['loss_fn']
    lr = train_configs['lr']
    beta1 = train_configs['beta1']
    beta2 = train_configs['beta2']
    checkpoint_factor = train_configs['save_checkpoint_steps']
    n_samples = train_configs['n_samples_to_generate']
    n_disc_updates = train_configs['n_disc_updates']
    display_step = train_configs['display_step']
    checkpoint_path = train_configs['train_from_checkpoint']

    # set loss function
    if loss_fn == "hinge":
        loss = losses.HingeGAN
    elif loss_fn == "relativistic-hinge":
        loss = losses.RelativisticAverageHingeGAN
    elif loss_fn == "standard-gan":
        loss = losses.StandardGAN
    elif loss_fn == "lsgan":
        loss = losses.LSGAN
    elif loss_fn == "lsgan-sigmoid":
        loss = losses.LSGAN_SIGMOID
    elif loss_fn == "wgan-gp":
        loss = losses.WGAN_GP
    else:
        raise Exception("Unknown loss function requested")

    # set optimizers
    gen_optim = torch.optim.Adam(model.gen.parameters(), lr=lr, betas=(beta1, beta2))
    disc_optim = torch.optim.Adam(model.dis.parameters(), lr=lr, betas=(beta1, beta2))

    if checkpoint_path is not None:
        model_state_dict = torch.load(checkpoint_path)
        start = model_state_dict["epoch"]
        model.gen.load_state_dict(model_state_dict["gen_state_dict"])
        model.dis.load_state_dict(model_state_dict["disc_state_dict"])
        gen_optim.load_state_dict(model_state_dict["gen_optim_state_dict"])
        disc_optim.load_state_dict(model_state_dict["disc_optim_state_dict"])
    else:
        start = 1

    # train the gan
    model.train(data_loader, gen_optim, disc_optim, n_disc_updates=n_disc_updates, loss_fn=loss(model.dis), num_epochs=epochs,
                checkpoint_factor=checkpoint_factor, num_samples=n_samples, display_step=display_step,
                save_dir="checkpoints/"+kwargs["checkpoints_fname"], start=start)

    return model


# def train(gen, disc, gen_loss_fn, disc_loss_fn, data_loader, device, *args, **kwargs):
#
#     # set optimizers
#     gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
#     disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
#
#     # records training steps
#     cur_step = 0
#
#     # saves generator and discriminator/critic losses
#     generator_losses = []
#     discriminator_losses = []
#
#     # Writes logs to TensorBoard
#     writer = SummaryWriter()
#
#     for epoch in range(epochs):
#         if verbose:
#             print("Epoch {}/{}".format(epoch + 1, epochs))
#
#         # updates the loss of gen and disc
#         mean_generator_loss = 0
#         mean_discriminator_loss = 0
#
#         for reals, _ in tqdm(data_loader):
#             cur_batch_size = len(reals)
#             reals = reals.to(device)
#
#             # average discriminator loss for the batch (averaged over number of disc updates)
#             mean_iteration_disc_loss = 0
#
#             # create a list of downsampled images from the real images:
#             real_downsamples = [reals] + [F.avg_pool2d(reals, int(np.power(2, i))) for i in range(1, gen.depth)]
#             real_downsamples = list(reversed(real_downsamples))
#
#             # update discriminator/critic
#             for _ in range(n_disc_updates):
#                 disc_opt.zero_grad()
#                 fake_noise = get_noise(cur_batch_size, z_dim, device=device)
#                 fakes = gen(fake_noise)
#                 fakes = [fake.detach() for fake in fakes]
#                 disc_fake_pred = disc(fakes)
#                 disc_real_pred = disc(real_downsamples)
#
#                 # compute discriminator/critic losses
#                 epsilon = torch.rand(cur_batch_size, 1, 1, 1, device=device, requires_grad=True)
#                 gradient = get_gradient_msggan(disc, real_downsamples, fakes, epsilon)
#                 gp = gradient_penalty(gradient)
#                 disc_loss = disc_loss_fn(disc_fake_pred, disc_real_pred, gp=gp, c_lambda=c_lambda)
#
#                 # keep track of the average discriminator loss in this batch
#                 mean_iteration_disc_loss += disc_loss.item() / n_disc_updates
#                 # update gradients
#                 disc_loss.backward(retain_graph=True)
#                 # update optimizer
#                 disc_opt.step()
#
#             # Keep track of average discriminator loss (weighted by batch-size)
#             mean_discriminator_loss += mean_iteration_disc_loss * (cur_batch_size / len(data_loader.dataset))
#             discriminator_losses.append(mean_discriminator_loss)
#
#             # Update generator
#             gen_opt.zero_grad()
#             fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
#             fakes_2 = gen(fake_noise_2)
#             disc_fake_pred = disc(fakes_2)
#             gen_loss = gen_loss_fn(disc_fake_pred, torch.ones_like(disc_fake_pred))
#             generator_losses.append(gen_loss.item())
#             gen_loss.backward()
#             gen_opt.step()
#
#             torch.cuda.empty_cache()
#
#             # Keep track of the average generator loss (weighted by batch-size)
#             mean_generator_loss += gen_loss.item() * (cur_batch_size / len(data_loader.dataset))
#             generator_losses.append(mean_generator_loss)
#
#             # Visualization code: observe real and fake images every 500 steps of training
#             if cur_step % display_step == 0 and cur_step > 0:
#                 real_images = show_tensor_images(reals)
#                 fake_images = show_tensor_images(fakes[-1])
#                 writer.add_image("Reals", real_images, cur_step, dataformats='CHW')
#                 writer.add_image("Fakes", fake_images, cur_step, dataformats='CHW')
#             cur_step += 1
#
#         # visualize generator and discriminator loss metrics per epoch
#         writer.add_scalars("Loss", {'Generator': mean_generator_loss,
#                                     'Discriminator': mean_discriminator_loss}, epoch)
#
#     writer.flush()
#     writer.close()
#
#     return gen, disc
