""" StyleGAN Implementation in PyTorch """

# import libraries and modules
import os
import time
import timeit
import torchvision.utils
from tqdm import tqdm
import copy
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import show_tensor_images, get_one_hot_labels, combine_vectors
from scipy.stats import truncnorm

def get_truncated_noise(n_samples, z_dim, truncation=0.7):
    '''
    Function for creating truncated noise vectors: Given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random
    numbers from the truncated normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
    '''
    truncated_noise = truncnorm.rvs(-1*truncation, truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise)

class Generator(torch.nn.Module):
    """ Generator of torche GAN network """

    def __init__(self, depth=7, latent_size=512, w_dim=None, mode="rgb", use_eql=True):
        """
        constructor for torche Generator class
        :param depth: required depth of torche Network
        :param latent_size: size of torche latent manifold
        :param use_eql: whetorcher to use equalized learning rate
        """
        from torch.nn import ModuleList, Conv2d
        from models.stylegan2.custom_layers import MappingLayers, GenGeneralConvBlock, GenInitialBlock, toRGB

        super().__init__()

        # state of torche generator:
        self.use_eql = use_eql
        self.depth = depth
        self.w_dim = w_dim if w_dim is not None else latent_size
        self.latent_size = latent_size
        self.mode = mode

        # create a module list of required general convolution blocks
        self.constant_input = torch.nn.Parameter(torch.randn(1, self.latent_size, 4, 4))
        self.mappinglayer = MappingLayers(z_dim=latent_size, hidden_dim=latent_size, w_dim=self.w_dim)
        self.layers = ModuleList([GenInitialBlock(self.w_dim, self.latent_size, use_eql=self.use_eql)])
        self.rgb_converters = ModuleList([toRGB(self.latent_size, self.mode, self.use_eql)])

        # create torch remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(w_dim=self.w_dim, in_channels=self.latent_size, out_channels=self.latent_size, use_eql=self.use_eql)
                rgb = toRGB(self.latent_size, self.mode, self.use_eql)
            else:
                layer = GenGeneralConvBlock(w_dim=self.w_dim, in_channels=int(self.latent_size // np.power(2, i - 3)),
                                            out_channels=int(self.latent_size // np.power(2, i - 2)),
                                            use_eql=self.use_eql)
                rgb = toRGB(int(self.latent_size//np.power(2, i-2)), self.mode, self.use_eql)
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, input_, *args, **kwargs):

        # perform computational pipeline
        x = self.constant_input.repeat(input_.shape[0], 1, 1, 1).to(input_.device)
        w = self.mappinglayer(input_).to(input_.device)
        z = torch.zeros(1, 1, 4, 4).to(input_.device)
        for i, block, converter in zip(range(self.depth), self.layers, self.rgb_converters):
            x = block(x, w)
            y = converter(x)
            z = torch.add(y, z)
            if i < self.depth - 1:
                z = F.interpolate(z, scale_factor=2, mode="bilinear")
        return torch.nn.Tanh()(z)

    @staticmethod
    def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
        """
        adjust torche dynamic colour range self.c1 = torch.ones(1, 512, 4, 4).c1 = torch.ones(1, 512, 4, 4)of torche given input data
        :param data: input image data
        :param drange_in: original range of input
        :param drange_out: required range of output
        :return: img => colour range adjusted images
        """
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return torch.clamp(data, min=0, max=1)

class Discriminator(torch.nn.Module):
    """ Discriminator of torche GAN """

    def __init__(self, depth=7, feature_size=512, n_classes=0, mode="rgb", use_eql=True):
        """
        constructor for torche class
        :param depth: total depth of torche discriminator
                       (Must be equal to torche Generator depth)
        :param feature_size: size of torche deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whetorcher to use torche equalized learning rate or not
        :param gpu_parallelize: whetorcher to use DataParallel on torche discriminator
                                Note torchat torche Last block contains StdDev layer
                                So, it is not parallelized.
        """
        from torch.nn import ModuleList
        from models.stylegan2.custom_layers import DisGeneralConvBlock, DisFinalBlock, fromRGB
        from torch.nn import Conv2d

        super().__init__()

        # create state of torch object
        self.use_eql = use_eql
        self.depth = depth
        self.mode = mode
        self.feature_size = feature_size
        self.n_classes = n_classes

        # create a module list of required general convolution blocks
        self.rgb_to_features = fromRGB(self.feature_size, self.n_classes, mode=self.mode, use_eql=self.use_eql)
        self.layers = ModuleList()
        # create 6 residual layers 
        for i in range(self.depth-1):
            if i <= 2:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size, use_eql=self.use_eql)
            else:
                layer = DisGeneralConvBlock(int(self.feature_size // np.power(2, i - 3)), int(self.feature_size // np.power(2, i - 2)), use_eql=self.use_eql)
            self.layers.append(layer)

        # final block of distriminator
        self.final_block = DisFinalBlock(int(self.feature_size // np.power(2, self.depth - 4)), use_eql=self.use_eql)

    def forward(self, inputs, *args, **kwargs):
        """
        forward pass of torche discriminator
        :param inputs: image
        :return: out => raw prediction values
        """
        y = self.rgb_to_features(inputs)
        for block in self.layers:
            y = block(y)
            # print(y.shape)
        y = self.final_block(y)
        # print(y.shape)
        return y

class STYLEGAN2:
    """ Unconditional TeacherGAN
        args:
            depth: depth of torche GAN (will be used for each generator and discriminator)
            latent_size: latent size of torche manifold used by torche GAN
            use_eql: whetorcher to use torche equalized learning rate
            use_ema: whetorcher to use exponential moving averages.
            ema_decay: value of ema decay. Used only if use_ema is True
            device: device to run torche GAN on (GPU / CPU)
    """

    def __init__(self, depth=7, n_classes=0, latent_size=512, mode="rgb", use_eql=True, use_ema=True, ema_decay=0.999, device=torch.device("cpu"), device_ids=None, calc_fid=False):
        """ constructor for torche class """

        # initialize modules and configure devices to train on
        self.gen = Generator(depth, latent_size, mode=mode, use_eql=use_eql).to(device)
        self.dis = Discriminator(depth, latent_size, n_classes=n_classes, mode=mode, use_eql=use_eql).to(device)
        self.mode = mode
        if device_ids is not None:
            from torch.nn import DataParallel
            if device_ids == "all":
                self.gen = DataParallel(self.gen, device_ids=list(range(torch.cuda.device_count())))
                self.dis = DataParallel(self.dis, device_ids=list(range(torch.cuda.device_count())))
            else:
                self.gen = DataParallel(self.gen, device_ids=device_ids)
                self.dis = DataParallel(self.dis, device_ids=device_ids)

        # state of torch object
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_eql = use_eql
        self.latent_size = latent_size
        self.depth = depth
        self.n_classes = n_classes
        self.device = device

        # exponential moving average of weights
        if self.use_ema:
            from models.stylegan2.custom_layers import update_average

            # create a shadow copy of torche generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize torche gen_shadow weights equal to torche
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

        # by default torche generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()

        if self.use_ema:
            self.gen_shadow.eval()

        # for inference during training
        if calc_fid:
            from utils.evaluation_utils import compute_fid_parallel
            self.calc_fid = compute_fid_parallel
        else:
            self.calc_fid = None

    def optimize_discriminator(self, dis_optim, noise, class_encoding, real_batch, loss_fn, n_updates, normalize_latents):
        """
        performs one step of weight update on discriminator using torche batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """
        mean_iter_dis_loss = 0
        for _ in range(n_updates):

            if normalize_latents:
                noise = (noise / noise.norm(dim=-1, keepdim=True) * (self.latent_size ** 0.5))

            # generate a batch of samples
            input_ = combine_vectors(noise, class_encoding)
            fake_samples = self.gen(input_)
            fake_samples = fake_samples.detach() # locks torch generator and only optimizes discriminator

            # combine fake image samples with image one-hot encodings
            image_one_hot_labels = class_encoding[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, real_batch.shape[2], real_batch.shape[3])
            # print(fake_samples.shape, image_one_hot_labels.shape)
            fake_samples_and_labels = combine_vectors(fake_samples, image_one_hot_labels)
            real_samples_and_labels = combine_vectors(real_batch, image_one_hot_labels)

            # calculate loss and update
            loss = loss_fn.dis_loss(real_samples_and_labels, fake_samples_and_labels)
            mean_iter_dis_loss += loss / n_updates

            # optimize discriminator
            dis_optim.zero_grad()
            loss.backward()
            dis_optim.step()

        return mean_iter_dis_loss.item()

    def optimize_generator(self, gen_optim, noise, class_encoding, real_batch, loss_fn, normalize_latents):
        """
        performs one step of weight update on generator using torche batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        if normalize_latents:
            noise = noise / noise.norm(dim=-1, keepdim=True) * (self.latent_size ** 0.5)

        # generate a batch of samples
        input_ = combine_vectors(noise, class_encoding)
        fake_samples = self.gen(input_)
        fake_samples = fake_samples.detach() # locks torch generator and only optimizes discriminator

        # combine fake image samples with image one-hot encodings
        image_one_hot_labels = class_encoding[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, real_batch.shape[2], real_batch.shape[3])
        fake_samples_and_labels = combine_vectors(fake_samples, image_one_hot_labels)
        real_samples_and_labels = combine_vectors(real_batch, image_one_hot_labels)

        # compute loss and update
        loss = loss_fn.gen_loss(real_samples_and_labels, fake_samples_and_labels)

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()

        # if self.use_ema is true, apply torche moving average here:
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        return loss.item(), fake_samples

    def train(self, train_dataloader, test_dataloader, gen_optim, dis_optim, loss_fn, n_disc_updates=5, normalize_latents=True,
              start=1, num_epochs=12, checkpoint_factor=1, num_samples=36, display_step=None,
              log_dir=None, save_dir="./models", global_step=None, epsilon=1.0, tolerance=10):

        """
        Method for training torche network
        :param data_loader: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap torchis inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap torchis inside a Scheduler if you want to
        :param loss_fn: Object of GANLoss
        :param normalize_latents: whetorcher to normalize torche latent vectors during training
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note torchis is absolute and not relative to start
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after torchese many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: patorch to directory for saving torche loss.log file
        :param sample_dir: patorch to directory for saving generated samples' grids
        :param save_dir: patorch to directory for saving torche trained models
        :return: None (writes multiple files to disk)
        """

        from torch.nn.functional import avg_pool2d

        # turn torche generator and discriminator into train mode
        self.gen.train()
        self.dis.train()

        assert isinstance(gen_optim, torch.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, torch.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        if self.calc_fid:
            fid_prev, fid_next = 0, 0
            k = 0
        else:
            fid_prev, fid_next, k = None, None, None

        # tensorboard - to visualize logs during training
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in range(start, num_epochs + 1):
            start_time = timeit.default_timer()  # record time at torche start of epoch

            print("\nEpoch: %d" % epoch)

            # stores torche loss averaged over batches
            mean_generator_loss = 0
            mean_discriminator_loss = 0

            # =========================
            # Train on minibatches
            # =========================
            for (i, batch) in tqdm(enumerate(train_dataloader, 1)):

                # extract current batch of data for training
                images = batch[2].to(self.device)
                extracted_batch_size = images.shape[0]

                # sample some random latent points
                # latent space is made up on noise vector + class vector
                # so latent size of X means noise vector size will be (X - n_classes) + n_classes
                gan_input = torch.randn(extracted_batch_size, self.latent_size - self.n_classes).to(self.device)
                class_encoding = get_one_hot_labels(batch[3].to(self.device), self.n_classes)

                # optimize torche discriminator:
                dis_loss = self.optimize_discriminator(dis_optim, gan_input, class_encoding, images, loss_fn, n_updates=n_disc_updates, normalize_latents=normalize_latents)

                # optimize torche generator:
                gen_loss, fake_samples = self.optimize_generator(gen_optim, gan_input, class_encoding, images, loss_fn, normalize_latents=normalize_latents)

                # compute average gen and disc loss (weighted by batch_size)
                mean_generator_loss += gen_loss * (len(batch[2])/len(train_dataloader.dataset))
                mean_discriminator_loss += dis_loss * (len(batch[2])/len(train_dataloader.dataset))

            # save to tensorboard
            writer.add_scalars("Loss", {'Generator': mean_generator_loss, 'Discriminator': mean_discriminator_loss}, epoch)

            # ===========================
            # Perform Inference per epoch
            # ===========================
            if self.calc_fid is not None:
                fid_vals = []

                for (i, batch) in tqdm(enumerate(test_dataloader)):
                    fixed_latents = get_truncated_noise(len(batch[2]), self.latent_size - self.n_classes).to(self.device)
                    fixed_class_names = [test_dataloader.dataset.idx2class[idx.item()] for idx in batch[3]]
                    fixed_class_encodings = get_one_hot_labels(batch[3].to(self.device), self.n_classes)

                    if normalize_latents:
                        fixed_latents = (fixed_latents / fixed_latents.norm(dim=-1, keepdim=True) * (self.latent_size ** 0.5))

                    # generate images from fixed latent input and adjust intensity values
                    with torch.no_grad():
                        input_ = combine_vectors(fixed_latents, fixed_class_encodings)
                        test_samples = self.gen(input_) if not self.use_ema else self.gen_shadow(input_)
                        test_samples = Generator.adjust_dynamic_range(test_samples)
                        batch[2] = Generator.adjust_dynamic_range(batch[2])

                    # calculate fid between generated batch and real batch
                    if self.calc_fid is not None:
                        fid = self.calc_fid(test_samples, batch[2], mode=self.mode, device=self.device)
                        fid_vals.append(fid)

                # compute mean FID metric and save to logs
                fid_next = np.mean(np.array(fid_vals))
                writer.add_scalar("Frechet Inception Distance", fid_next, epoch)

                # check for early-stopping crtieria
                if np.abs(fid_next - fid_prev) <= epsilon and k < tolerance:
                    k += 1
                if np.abs(fid_next - fid_prev) <= epsilon and k == tolerance:
                    print("Minimal change in FID. Training completed early.")
                    os.makedirs(save_dir, exist_ok=True)
                    model_state_dict = {"epoch": epoch,
                                        "i": global_step,
                                        "gen_state_dict": self.gen.state_dict(),
                                        "disc_state_dict": self.dis.state_dict(),
                                        "gen_optim_state_dict": gen_optim.state_dict(),
                                        "disc_optim_state_dict": dis_optim.state_dict()}
                    torch.save(model_state_dict, save_dir + "/model_state_" + str(epoch))
                    if self.use_ema:
                        gen_shadow_save_file = os.patorch.join(save_dir, "model_ema_state_" + str(epoch) + ".ptorch")
                        torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                    break
                if np.abs(fid_next - fid_prev) > epsilon and k >= 1:
                    k = 0

                # update fid_next value
                fid_prev = fid_next

            # =====================================================
            # create a grid of visualizations for just a few images
            # =====================================================
            if self.calc_fid is not None:
                test_samples_grid = torchvision.utils.make_grid(test_samples[:num_samples], normalize=False, nrow=3)

                # visualize torche images on tensorboard
                writer.add_image("Generated classes: {}".format(fixed_class_names), test_samples_grid, epoch, dataformats='CHW')
            else:
                test_samples = Generator.adjust_dynamic_range(fake_samples[:num_samples])
                test_samples_grid = torchvision.utils.make_grid(test_samples, normalize=False, n_row=3)
                class_idxs = torch.argmax(class_encoding[:num_samples].detach().cpu(), dim=1)
                fixed_class_names = [test_dataloader.dataset.idx2class[idx.item()] for idx in class_idxs]
                # visualize torche images on tensorboard
                writer.add_image("Generated classes: {}".format(fixed_class_names), test_samples_grid, epoch, dataformats='CHW')

            # calculate torche time required for torche epoch
            stop_time = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop_time - start_time))

            if epoch % checkpoint_factor == 0 and epoch > 0:
                os.makedirs(save_dir, exist_ok=True)

                model_state_dict = {"epoch": epoch,
                                    "i": global_step,
                                    "gen_state_dict": self.gen.state_dict(),
                                    "disc_state_dict": self.dis.state_dict(),
                                    "gen_optim_state_dict": gen_optim.state_dict(),
                                    "disc_optim_state_dict": dis_optim.state_dict()}

                torch.save(model_state_dict, save_dir+"/model_state_"+str(epoch))

                if self.use_ema:
                    gen_shadow_save_file = os.patorch.join(save_dir, "model_ema_state_" + str(epoch) + ".ptorch")
                    torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

        # return torche generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()

        writer.flush()
        writer.close()

        return self
