""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
    Code taken from -> https://github.com/akanimax/BMSG-GAN
"""

# import libraries and modules
import datetime
import os
import time
import timeit

import torchvision.utils
from tqdm import tqdm
import copy
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import show_tensor_images


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, mode="rgb", use_eql=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, Conv2d
        from models.msggan.custom_layers import GenGeneralConvBlock, \
            GenInitialBlock, _equalized_conv2d

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size
        self.mode = mode

        # register the modules required for the Generator Below ...
        # create the ToRGB layers for various outputs:
        if self.use_eql:
            def to_rgb(in_channels):
                if self.mode == "rgb":
                    return _equalized_conv2d(in_channels, 3, (1, 1), bias=True)
                elif self.mode == "grayscale":
                    return _equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            def to_rgb(in_channels):
                if self.mode == "rgb":
                    return Conv2d(in_channels, 3, (1, 1), bias=True)
                elif self.mode == "grayscale":
                    return Conv2d(in_channels, 1, (1, 1), bias=True)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([GenInitialBlock(self.latent_size, use_eql=self.use_eql)])
        self.rgb_converters = ModuleList([to_rgb(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size,
                                            use_eql=self.use_eql)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, x, *args, **kwargs):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        outputs = []  # initialize to empty list

        y = x  # start the computational pipeline
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(converter(y))

        return outputs

    @staticmethod
    def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
        """
        adjust the dynamic colour range of the given input data
        :param data: input image data
        :param drange_in: original range of input
        :param drange_out: required range of output
        :return: img => colour range adjusted images
        """
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return th.clamp(data, min=0, max=1)


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512, mode="rgb",
                 use_eql=True, gpu_parallelize=False):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use the equalized learning rate or not
        :param gpu_parallelize: whether to use DataParallel on the discriminator
                                Note that the Last block contains StdDev layer
                                So, it is not parallelized.
        """
        from torch.nn import ModuleList
        from models.msggan.custom_layers import DisGeneralConvBlock, \
            DisFinalBlock, _equalized_conv2d
        from torch.nn import Conv2d

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.gpu_parallelize = gpu_parallelize
        self.use_eql = use_eql
        self.depth = depth
        self.mode = mode
        self.feature_size = feature_size

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            def from_rgb(out_channels):
                if self.mode == "rgb":
                    return _equalized_conv2d(3, out_channels, (1, 1), bias=True)
                elif self.mode == "grayscale":
                    return _equalized_conv2d(1, out_channels, (1, 1), bias=True)
        else:
            def from_rgb(out_channels):
                if self.mode == "rgb":
                    return Conv2d(3, out_channels, (1, 1), bias=True)
                elif self.mode == "grayscale":
                    return Conv2d(1, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList()
        self.final_converter = from_rgb(self.feature_size // 2)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList()
        self.final_block = DisFinalBlock(self.feature_size, use_eql=self.use_eql)

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            use_eql=self.use_eql)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 2] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # parallelize the modules from the module-lists if asked to:
        if self.gpu_parallelize:
            for i in range(len(self.layers)):
                self.layers[i] = th.nn.DataParallel(self.layers[i])
                self.rgb_to_features[i] = th.nn.DataParallel(
                    self.rgb_to_features[i])

        # Note that since the FinalBlock contains the StdDev layer,
        # it cannot be parallelized so easily. It will have to be parallelized
        # from the Lower level (from CustomLayers). This much parallelism
        # seems enough for me.

    def forward(self, inputs, *args, **kwargs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 2](inputs[self.depth - 1])
        y = self.layers[self.depth - 2](y)
        for x, block, converter in \
                zip(reversed(inputs[1:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = th.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        # calculate the final block:
        input_part = self.final_converter(inputs[0])
        y = th.cat((input_part, y), dim=1)
        y = self.final_block(y)

        # return calculated y
        return y


class MSG_GAN:
    """ Unconditional TeacherGAN
        args:
            depth: depth of the GAN (will be used for each generator and discriminator)
            latent_size: latent size of the manifold used by the GAN
            use_eql: whether to use the equalized learning rate
            use_ema: whether to use exponential moving averages.
            ema_decay: value of ema decay. Used only if use_ema is True
            device: device to run the GAN on (GPU / CPU)
    """

    def __init__(self, depth=7, latent_size=512, mode="rgb",
                 use_eql=True, use_ema=True, ema_decay=0.999,
                 device=th.device("cpu"), device_ids=None, calc_fid=False):
        """ constructor for the class """

        self.gen = Generator(depth, latent_size, mode=mode, use_eql=use_eql).to(device)
        self.dis = Discriminator(depth, latent_size, mode=mode, use_eql=True).to(device)
        self.mode = mode
        if device_ids is not None:
            from torch.nn import DataParallel
            if device_ids == "all":
                self.gen = DataParallel(self.gen, device_ids=list(range(th.cuda.device_count())))
                self.dis = DataParallel(self.dis, device_ids=list(range(th.cuda.device_count())))
            else:
                self.gen = DataParallel(self.gen, device_ids=device_ids)
                self.dis = DataParallel(self.dis, device_ids=device_ids)

        # state of the object
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_eql = use_eql
        self.latent_size = latent_size
        self.depth = depth
        self.device = device

        if self.use_ema:
            from models.msggan.custom_layers import update_average

            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()

        if calc_fid:
            from utils.evaluation_utils import compute_fid_parallel
            self.calc_fid = compute_fid_parallel
        else:
            self.calc_fid = None

    def generate_samples(self, num_samples):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated
        :return: generated samples tensor: list[ Tensor(B x H x W x C)]
        """
        noise = th.randn(num_samples, self.latent_size).to(self.device)
        generated_images = self.gen(noise)

        # reshape the generated images
        generated_images = list(map(lambda x: (x.detach().permute(0, 2, 3, 1) / 2) + 0.5,
                                    generated_images))

        return generated_images

    def optimize_discriminator(self, dis_optim, noise, real_batch, loss_fn, n_updates):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """
        mean_iter_dis_loss = 0
        for _ in range(n_updates):
            # generate a batch of samples
            fake_samples = self.gen(noise)
            fake_samples = list(map(lambda x: x.detach(), fake_samples))

            loss = loss_fn.dis_loss(real_batch, fake_samples)

            mean_iter_dis_loss += loss / n_updates

            # optimize discriminator
            dis_optim.zero_grad()
            loss.backward()
            dis_optim.step()

        return mean_iter_dis_loss.item()

    def optimize_generator(self, gen_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)

        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()

        # if self.use_ema is true, apply the moving average here:
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        return loss.item(), fake_samples

    def create_grid(self, images, n_samples_visualize):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing list[Tensors]
        :param img_files: list of names of files to write
        :return: None (saves multiple files)
        """
        from torch.nn.functional import interpolate
        from numpy import power

        # resize the samples to have same resolution:
        for i in range(len(images)):
            images[i] = interpolate(images[i][:n_samples_visualize], scale_factor=power(2, self.depth - 1 - i))

        # reshape into grid format
        images_tensor = th.cat(images, 0)
        images_grid = show_tensor_images(images_tensor, normalize=False, n_rows=n_samples_visualize)

        return images_grid

    def train(self, train_data_loader, test_data_loader, gen_optim, dis_optim, loss_fn, n_disc_updates=5, normalize_latents=True,
              start=1, num_epochs=12, checkpoint_factor=1, num_samples=36, display_step=None,
              log_dir=None, save_dir="./models", global_step=None, epsilon=3.0, tolerance=10):

        """
        Method for training the network
        :param data_loader: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap this inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap this inside a Scheduler if you want to
        :param loss_fn: Object of GANLoss
        :param normalize_latents: whether to normalize the latent vectors during training
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note this is absolute and not relative to start
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after these many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: path to directory for saving the loss.log file
        :param sample_dir: path to directory for saving generated samples' grids
        :param save_dir: path to directory for saving the trained models
        :return: None (writes multiple files to disk)
        """

        from torch.nn.functional import avg_pool2d

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()

        assert isinstance(gen_optim, th.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, th.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        if self.calc_fid:
            fid_prev, fid_next = 0, 0
            k = 0
        else:
            fid_prev, fid_next, k = None, None, None

        # tensorboard - to visualize logs during training
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in range(start, num_epochs + 1):
            start_time = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)

            # stores the loss averaged over batches
            mean_generator_loss = 0
            mean_discriminator_loss = 0

            # ==============================
            # Train on minibatches
            # ==============================

            for (i, batch) in tqdm(enumerate(train_data_loader, 1)):

                # extract current batch of data for training
                images = batch[2].to(self.device)
                extracted_batch_size = images.shape[0]

                # create a list of downsampled images from the real images:
                images = [images] + [avg_pool2d(images, int(np.power(2, i))) for i in range(1, self.depth)]
                images = list(reversed(images))

                # sample some random latent points
                gan_input = th.randn(extracted_batch_size, self.latent_size).to(self.device)

                # normalize them if asked
                if normalize_latents:
                    gan_input = (gan_input
                                 / gan_input.norm(dim=-1, keepdim=True)
                                 * (self.latent_size ** 0.5))

                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(dis_optim, gan_input,
                                                       images, loss_fn, n_updates=n_disc_updates)

                # optimize the generator:
                gen_loss, fake_samples = self.optimize_generator(gen_optim, gan_input, images, loss_fn)

                # compute average gen and disc loss (weighted by batch_size)
                mean_generator_loss += gen_loss * (len(batch)/len(train_data_loader.dataset))
                mean_discriminator_loss +=  dis_loss * (len(batch)/len(train_data_loader.dataset))

            # save to tensorboard
            writer.add_scalars("Loss", {"Generator": mean_generator_loss,
                                        "Discriminator": mean_discriminator_loss}, epoch)

            # ==============================
            # Evaluate per epoch
            # ==============================
            if self.calc_fid is not None:
                fid_vals = []

            for (i, batch) in tqdm(enumerate(test_data_loader)):
                fixed_latents = th.randn(len(batch[2]), self.latent_size).to(self.device)

                if normalize_latents:
                    fixed_latents = (fixed_latents
                                   / fixed_latents.norm(dim=-1, keepdim=True)
                                   * (self.latent_size ** 0.5))

                # generate images from fixed latent input and adjust intensity values
                with th.no_grad():
                    test_samples = self.gen(fixed_latents) if not self.use_ema else self.gen_shadow(fixed_latents)
                    test_samples = [Generator.adjust_dynamic_range(sample) for sample in test_samples]

                # calculate fid between generated batch and real batch
                if self.calc_fid is not None:
                    fid = self.calc_fid(test_samples[-1], batch[2], mode=self.mode, device=self.device)
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
                th.save(model_state_dict, save_dir + "/model_state_" + str(epoch))
                if self.use_ema:
                    gen_shadow_save_file = os.path.join(save_dir, "model_ema_state_" + str(epoch) + ".pth")
                    th.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                break
            if np.abs(fid_next - fid_prev) > epsilon and k >= 1:
                k = 0

            # update fid_next value
            fid_prev = fid_next

            # =====================================================
            # create a grid of visualizations for just a few images
            # =====================================================
            test_samples_grid = torchvision.utils.make_grid(test_samples[-1][:num_samples], normalize=False, nrow=3)

            # visualize the images on tensorboard
            writer.add_image("Synthetic Images", test_samples_grid, epoch, dataformats='CHW')

            # calculate the time required for the epoch
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

                th.save(model_state_dict, save_dir + "/model_state_" + str(epoch))

                if self.use_ema:
                    gen_shadow_save_file = os.path.join(save_dir, "model_ema_state_" + str(epoch) + ".pth")
                    th.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()

        writer.flush()
        writer.close()

        return self
