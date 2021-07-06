""" TRAINER SCRIPT FOR PROGRESSIVE GANS MODEL """

from datetime import datetime
import yaml
import copy
from torchvision import transforms
from helpers.data import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def filename(configs):
    """ Creates a readable filename convention for trained models """
    dataset_name = "demo" if configs["data_directory"] == "demo" else \
        os.path.basename(os.path.normpath(configs["data_directory"]))
    file_name = "data:{}_trans:{}-{}-{}_mod:{}-{}-{}-{}-{}_tr:{}-{}-{}-{}-{}".format(dataset_name,
                                                                                     configs["transformations"]["resize_dim"],
                                                                                     configs["transformations"]["grayscale"],
                                                                                     configs["transformations"]["normalize"],
                                                                                     configs["model"],
                                                                                     configs["z_dim"],
                                                                                     configs['c_lambda'],
                                                                                     configs['n_disc_updates'],
                                                                                     configs['max_resolution'],
                                                                                     configs['epochs'],
                                                                                     configs["batch_size"],
                                                                                     configs["lr"],
                                                                                     configs["beta1"],
                                                                                     configs["beta2"])
    return file_name


def load_config(config_file):
    """ Loads model configuration file """
    with open(config_file) as file:
        conf = yaml.safe_load(file)
    return conf


def create_dataloaders(image_size, batch_size):
    """ Creates new dataloaders with modified image sizes """
    # transform the image data
    transformations = [transforms.Resize((image_size, image_size))]
    if config['transformations']['grayscale']:
        transformations.append(transforms.Grayscale())
    transformations.append(transforms.ToTensor())
    if config['transformations']['normalize']:
        if config['transformations']['grayscale']:
            transformations.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    # load as pytorch dataset
    images_dataset = ImageDataset(images_directory=config['data_directory'],
                                  transforms=transforms.Compose(transformations))
    # add image resizing
    dataloader = DataLoader(images_dataset, batch_size, shuffle=True)
    return dataloader


def accumulate(model1, model2, decay=0.999):
    """ Performs exponential moving average of weights for visualizing generator output during training """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        # par1[k] = decay*par1[k].data + (1-decay)*par2[k].data
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1-decay)


def save_checkpoint(gen_model, disc_model, gen_optim, disc_optim, epoch, depth, alpha):
    """ Creates a checkpoint file for the GAN """
    now = datetime.now()
    # create checkpoints directory
    if not os.path.exists("checkpoints/"):
        os.mkdir("checkpoints/")

    # create checkpoints folder for specific model
    model_folder = "checkpoints/"+filename(config) + "/"
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    PATH = model_folder + now.strftime("checkpoint_%d.%m.%Y;%H:%M:%S") + ".pt"
    # stores all the weights and states of the model
    model_state_dict = {"epoch": epoch,
                        "depth": depth,
                        "alpha": alpha,
                        "gen": {"state_dict": gen_model.state_dict(), "optim_state_dict": gen_optim.state_dict()},
                        "disc": {"state_dict": disc_model.state_dict(), "optim_state_dict": disc_optim.state_dict()}}
    torch.save(model_state_dict, PATH)


def load_from_checkpoint(path):
    """ Load a model from checkpoint path """
    checkpoint = torch.load(path)
    return checkpoint


# load configs file
CONFIG_PATH = "model_configs.yaml"
config = load_config(CONFIG_PATH)

# setup other training parameters
z_dim = config['z_dim']
display_step = config['display_step']
beta_1 = config['beta1']
beta_2 = config['beta2']
lr = config['lr']
# this is required for the wgan-gp loss
c_lambda = config['c_lambda']
n_disc_updates = config['n_disc_updates']
verbose = config['verbose']
max_resolution = config['max_resolution']
epochs = config['epochs']


def train(gen, disc, gen_loss_fn, disc_loss_fn, dataloader, device=None, train_from_checkpoint=None, *args, **kwargs):

    # sets the maximum depth to train a network to
    resolution_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    batch_sizes_per_resolution = [16, 16, 16, 16, 16, 16, 6, 6, 3] # reduce batch size of 256 > 6
    max_depth = resolution_list.index(max_resolution)

    # configure a separate generator model - will use this to visualize generator output during training
    gen_running = copy.deepcopy(gen)
    gen_running.train(False)

    # GAN optimizers
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_2, beta_2))

    # save generator and discriminator/critic losses here
    generator_losses = []
    discriminator_losses = []

    # stores number of training blocks
    # depth = 0
    # gen.res_depth = disc.res_depth = depth

    # counter to monitor when to display images
    cur_step = 0
    # counter to monitor when to phase in a new block
    # iteration = 0

    # writes logs to tensorboard
    writer = SummaryWriter()

    # skip iterations if we are loading a model from checkpoint
    if train_from_checkpoint:
        # load checkpoint dictionary
        checkpoint = load_from_checkpoint(train_from_checkpoint)
        # depth = checkpoint['depth']
        # iteration = checkpoint['iteration']
        start_i = checkpoint['epoch'] # depth * epochs + checkpoint[iteration]

        # set alpha and depth of the gen and disc
        gen.res_depth = disc.res_depth = checkpoint['depth']
        gen.alpha = disc.alpha = checkpoint['alpha']

        # load weights and optimizer states
        gen.load_state_dict(checkpoint['gen']['state_dict'])
        gen_opt.load_state_dict(checkpoint['gen']['optim_state_dict'])
        disc.load_state_dict(checkpoint['disc']['state_dict'])
        disc_opt.load_state_dict(checkpoint['disc']['optim_state_dict'])

        # configure a new dataloader according to the checkpoint
        dataloader = create_dataloaders(image_size=4 * 2 ** gen.res_depth,
                                        batch_size=batch_sizes_per_resolution[gen.res_depth])
    else:
        start_i = 0

    for i in range(start_i, (max_depth+1)*epochs):

        if verbose:
            if train_from_checkpoint:
                print("Resuming model training from latest checkpoint...")
            print("Iteration {}/{}".format(i+1, (max_depth+1)*epochs))

        # update alpha slowly with every iteration
        # gen.alpha = disc.alpha = iteration/(epochs - 1) # min(1.0, 0.00002 * iteration)

        # every 100000 iterations, we will "phase"-in a new block
        # if iteration == epochs:
        #     # restart parameters
        #     gen.alpha = disc.alpha = 0
        #     iteration = 0
        #     # phase-in next block
        #     depth += 1
        #     gen.res_depth = disc.res_depth = depth
        #     if depth > max_depth:
        #         gen.alpha = disc.alpha = 1
        #         gen.res_depth = disc.res_depth = max_depth
        #     # create new dataloader with modified batch and image sizes
        #     dataloader = create_dataloaders(image_size=4*2**depth,
        #                                     batch_size=batch_sizes_per_resolution[depth])

        # updates the losses of generator and discriminator
        mean_generator_loss = 0
        mean_discriminator_loss = 0

        # grabs a batch of examples
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

                # compute discriminator/critic loss
                epsilon = torch.rand(len(reals), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(disc, reals, fakes.detach(), epsilon)
                gp = gradient_penalty(gradient)
                disc_loss = disc_loss_fn(disc_fake_pred, disc_real_pred, gp=gp, c_lambda=c_lambda)

                # keep track of the average discriminator loss in this batch
                mean_iteration_disc_loss += disc_loss.item() / n_disc_updates
                # update gradients
                disc_loss.backward(retain_graph=True)
                # update optimizer
                disc_opt.step()

            # keep track of average discriminator loss (weighted by batch size)
            mean_discriminator_loss += mean_iteration_disc_loss * (cur_batch_size / len(dataloader.dataset))
            discriminator_losses.append(mean_discriminator_loss)

            # update generator
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
            fakes_2 = gen(fake_noise_2)
            disc_fake_pred = disc(fakes_2)
            gen_loss = gen_loss_fn(disc_fake_pred, torch.ones_like(disc_fake_pred))
            generator_losses.append(gen_loss.item())
            gen_loss.backward()
            gen_opt.step()

            # calculate exponential average of weights for visualizing
            accumulate(gen_running, gen)

            # free memory
            torch.cuda.empty_cache()

            # keep track of the average generator loss (weighted by batch size)
            mean_generator_loss += gen_loss.item() * (cur_batch_size / len(dataloader.dataset))
            generator_losses.append(mean_generator_loss)

            # visualize generator output
            if cur_step % display_step == 0 and cur_step > 0:
                real_images = show_tensor_images(reals)
                gen_running.res_depth = gen.res_depth
                gen_running.alpha = gen.alpha
                fakes = gen_running(get_noise(cur_batch_size, z_dim, device))
                fake_images = show_tensor_images(fakes)
                writer.add_image("Reals", real_images, cur_step, dataformats='CHW')
                writer.add_image("Fakes", fake_images, cur_step, dataformats='CHW')

            # increment counter
            cur_step += 1
        # iteration += 1

        # save model checkpoint
        if config['save_checkpoints']:
            if (i+1) % config['save_checkpoints'] == 0:
                save_checkpoint(gen, disc, gen_opt, disc_opt, epoch=i, depth=gen.res_depth, alpha=gen.alpha)

        # visualize generator and discriminator/critic losses
        writer.add_scalars("Loss", {"Generator": mean_generator_loss,
                                    "Discriminator": mean_discriminator_loss}, i)

        # if the block has been trained for X epochs, reset alpha to 0 and phase in next block
        if (i+1) % epochs == 0:
            # restart parameters
            gen.alpha = disc.alpha = 0
            # iteration = 0
            # phase-in next block
            # depth += 1
            gen.res_depth += 1
            disc.res_depth += 1
            if gen.res_depth > max_depth:
                gen.alpha = disc.alpha = 1
                gen.res_depth = disc.res_depth = max_depth
            # create new dataloader with modified batch and image sizes
            dataloader = create_dataloaders(image_size=4 * 2 ** gen.res_depth,
                                            batch_size=batch_sizes_per_resolution[gen.res_depth])
        # otherwise, just update the alpha
        else:
            gen.alpha += 1/(epochs - 1)
            disc.alpha += 1/(epochs - 1)

    writer.flush()
    writer.close()

    return gen, disc
