""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
    Code taken from -> https://github.com/akanimax/BMSG-GAN
"""

# import libraries and modules
import os
import sys
import time
import timeit

import torch.cuda
import torchvision.utils
from tqdm import tqdm
import copy
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
from helpers.data_utils import show_tensor_images, get_one_hot_labels, combine_vectors


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, n_classes=10, mode="rgb", use_eql=True):
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

        # assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
        #     "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.n_classes = n_classes
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size #+ 50
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

        # embedding layer to create label representations
        self.embedding_layer = th.nn.Embedding(n_classes, 50)
        # fully connected layers for processing label embeddings
        self.fc_layers = th.nn.ModuleList()
        for i in range(self.depth):
            res = np.power(2, i + 2)
            dense_layer = th.nn.Linear(50, res**2)
            self.fc_layers.append(dense_layer)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([GenInitialBlock(self.latent_size, use_eql=self.use_eql)])
        self.rgb_converters = ModuleList([to_rgb(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size+1, self.latent_size, use_eql=self.use_eql)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3))+1,
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, x, in_labels):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        outputs = []  # initialize to empty list
        labels_embedding = self.embedding_layer(in_labels)

        # y = th.cat([x, labels_embedding], dim=1)  # start the computational pipeline
        y = x
        for dense, block, converter in zip(self.fc_layers, self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(converter(y))
            class_embedding = dense(labels_embedding).view(len(in_labels), 1, y.shape[-2], y.shape[-1])
            y = th.cat([y, class_embedding], dim=1)

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

    def __init__(self, depth=7, feature_size=512, n_classes=10, mode="rgb",
                 use_eql=True):
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
        from torch.nn import Conv2d, Embedding

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.use_eql = use_eql
        self.depth = depth
        self.mode = mode
        self.feature_size = feature_size

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            def from_rgb(out_channels):
                if self.mode == "rgb":
                    return _equalized_conv2d(3+1, out_channels, (1, 1), bias=True)
                elif self.mode == "grayscale":
                    return _equalized_conv2d(1+1, out_channels, (1, 1), bias=True)
        else:
            def from_rgb(out_channels):
                if self.mode == "rgb":
                    return Conv2d(3+1, out_channels, (1, 1), bias=True)
                elif self.mode == "grayscale":
                    return Conv2d(1+1, out_channels, (1, 1), bias=True)

        self.n_classes = n_classes
        # embedding for learning class representations
        self.embedding_layer = Embedding(num_embeddings=n_classes, embedding_dim=50)
        # linear layer which maps embedding to (resolution**2, 1)
        self.fc_layers = ModuleList()

        for i in range(self.depth):
            res = np.power(2, i + 2)
            dense_layer = th.nn.Linear(50, res**2)
            self.fc_layers.append(dense_layer)

        # converts the rgb images of each resolution from the generator into image with multiple feature channels
        self.rgb_to_features = ModuleList()
        # final rgb to features converter
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
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2, use_eql=self.use_eql)
                rgb = from_rgb(self.feature_size // 2)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 2] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # parallelize the modules from the module-lists if asked to:
        # if self.gpu_parallelize:
        #     for i in range(len(self.layers)):
        #         self.layers[i] = th.nn.DataParallel(self.layers[i])
        #         self.rgb_to_features[i] = th.nn.DataParallel(self.rgb_to_features[i])

        # Note that since the FinalBlock contains the StdDev layer,
        # it cannot be parallelized so easily. It will have to be parallelized
        # from the Lower level (from CustomLayers). This much parallelism
        # seems enough for me.

    def forward(self, in_images, in_label):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """

        assert len(in_images) == self.depth, "Mismatch between input and Network scales"

        # convert integer class into a n-dimensional embedding vector
        label_embedding = self.embedding_layer(in_label)
        class_matrix = self.fc_layers[self.depth - 1](label_embedding)
        class_matrix = class_matrix.view(len(in_label), 1,
                                         in_images[self.depth - 1].shape[-2], in_images[self.depth - 1].shape[-1])

        # combine the last resolution image with its class matrix and pass through rgb_to_features
        y = self.rgb_to_features[self.depth - 2](th.cat([in_images[self.depth - 1], class_matrix], dim=1))
        y = self.layers[self.depth - 2](y)

        for x, dense, block, converter in zip(reversed(in_images[1:-1]),
                                              reversed(self.fc_layers[1:-1]),
                                              reversed(self.layers[:-1]),
                                              reversed(self.rgb_to_features[:-1])):

            class_matrix = dense(label_embedding).view(len(in_label), 1, x.shape[-2], x.shape[-1])
            input_part = converter(th.cat([x, class_matrix], dim=1))  # convert the input:
            y = th.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        # calculate the final block:
        class_matrix = self.fc_layers[0](label_embedding).view(len(in_label), 1,
                                                               in_images[0].shape[-2], in_images[0].shape[-1])
        input_part = self.final_converter(th.cat([in_images[0], class_matrix], dim=1))
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

    def __init__(self, depth=7, n_classes=10, latent_size=512, mode="rgb",
                 use_eql=True, use_ema=True, ema_decay=0.999,
                 device=th.device("cpu"), device_ids=None, calc_fid=False):

        """ constructor for the class """

        self.gen = Generator(depth, latent_size, n_classes=n_classes, mode=mode, use_eql=use_eql).to(device)
        self.dis = Discriminator(depth, latent_size, n_classes=n_classes, mode=mode, use_eql=True).to(device)

        # Parallelize them if required:
        if device_ids is not None:
            from torch.nn import DataParallel
            if device_ids == "all":
                self.gen = DataParallel(self.gen, device_ids=torch.cuda.device_count())
                self.dis = DataParallel(self.dis, device_ids=torch.cuda.device_count())
            else:
                self.gen = DataParallel(self.gen, device_ids=device_ids)
                self.dis = DataParallel(self.dis, device_ids=device_ids)

        # state of the object
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_eql = use_eql
        self.latent_size = latent_size
        self.depth = depth
        self.n_classes = n_classes
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
            from helpers.evaluate import compute_fid
            self.calc_fid = compute_fid
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

    def optimize_discriminator(self, dis_optim, noise, labels, real_batch, loss_fn, n_updates):
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
            fake_samples = self.gen(noise, labels)
            fake_samples = list(map(lambda x: x.detach(), fake_samples))

            # calculate loss and update
            loss = loss_fn.dis_loss(real_batch, fake_samples, labels)
            mean_iter_dis_loss += loss / n_updates

            # optimize discriminator
            dis_optim.zero_grad()
            loss.backward()
            dis_optim.step()

        return mean_iter_dis_loss.item()

    def optimize_generator(self, gen_optim, noise, labels, real_batch, loss_fn):
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
        fake_samples = self.gen(noise, labels)

        # calculate loss between real and fake batch
        loss = loss_fn.gen_loss(real_batch, fake_samples, labels)

        # calculate fid between real and fake batch
        # fid = self.calc_fid(fake_samples[-1].squeeze(), real_batch[-1].squeeze(), device=self.device)
        # if self.calc_fid is not None else None

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()

        # if self.use_ema is true, apply the moving average here:
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        return loss.item(), fake_samples

    def create_grid(self, samples, img_files=None):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing list[Tensors]
        :param img_files: list of names of files to write
        :return: None (saves multiple files)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate
        from numpy import sqrt, power

        # dynamically adjust the colour of the images
        samples = [Generator.adjust_dynamic_range(sample) for sample in samples]

        # resize the samples to have same resolution:
        for i in range(len(samples)):
            samples[i] = interpolate(samples[i], scale_factor=power(2, self.depth - 1 - i))

        # save the images:
        # for sample, img_file in zip(samples, img_files):
        #     save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])),
        #                normalize=True, scale_each=True, padding=0)

        return samples

    def train(self, data_loader, gen_optim, dis_optim, loss_fn, n_disc_updates=5, normalize_latents=True,
              start=1, num_epochs=12, feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=36, display_step=None,
              log_dir=None, sample_dir="./samples",
              save_dir="./models", global_step=None):

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

        # create fixed_input for debugging
        fixed_input = th.randn(num_samples, self.latent_size).to(self.device)
        fixed_labels = th.randint(low=0, high=self.n_classes, size=(num_samples,), device=self.device)
        fixed_genes = [data_loader.dataset.idx2class[idx.item()] for idx in fixed_labels]

        if normalize_latents:
            fixed_input = (fixed_input / fixed_input.norm(dim=-1, keepdim=True) * (self.latent_size ** 0.5))

        # create a global time counter
        global_time = time.time()
        if global_step is None:
            global_step = 0

        # store generator and discriminator losses
        generator_losses = []
        discriminator_losses = []

        # tensorboard - to visualize logs during training
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in range(start, num_epochs + 1):
            start_time = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)

            # stores the loss averaged over batches
            mean_generator_loss = 0
            mean_discriminator_loss = 0
            # if self.calc_fid is not None:
            #     mean_fid_score = 0

            for (i, batch) in tqdm(enumerate(data_loader, 1)):

                # extract current batch of data for training
                images = batch[0].to(self.device)
                extracted_batch_size = images.shape[0]

                # create a list of downsampled images from the real images:
                images = [images] + [avg_pool2d(images, int(np.power(2, i))) for i in range(1, self.depth)]
                images = list(reversed(images))

                # sample some random latent points
                gan_input = th.randn(extracted_batch_size, self.latent_size).to(self.device)
                labels = batch[1].to(self.device)

                # normalize them if asked
                if normalize_latents:
                    gan_input = (gan_input
                                 / gan_input.norm(dim=-1, keepdim=True)
                                 * (self.latent_size ** 0.5))

                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(dis_optim, gan_input, labels,
                                                       images, loss_fn, n_updates=n_disc_updates)

                # optimize the generator:
                gen_loss, fake_samples = self.optimize_generator(gen_optim, gan_input, labels, images, loss_fn)

                # compute average gen and disc loss (weighted by batch_size)
                mean_generator_loss += gen_loss * (len(batch)/len(data_loader.dataset))
                mean_discriminator_loss +=  dis_loss * (len(batch)/len(data_loader.dataset))
                # if self.calc_fid is not None:
                #     mean_fid_score += fid_score * (len(batch)/len(data_loader.dataset))

                if global_step % display_step == 0 and global_step > 0:
                    with th.no_grad():
                        samples = self.create_grid(self.gen(fixed_input, fixed_labels) if not self.use_ema
                                                   else self.gen_shadow(fixed_input, fixed_labels))

                    # combine all images into a single tensor
                    fake_images_tensor = th.cat(samples, 0)
                    fake_grid = show_tensor_images(fake_images_tensor, normalize=False, n_rows=len(samples[0]))
                    # visualize the images on tensorboard
                    writer.add_image("Generated classes: {}".format(fixed_genes), fake_grid, global_step, dataformats='CHW')

                    # calculate fid between generated batch and real batch
                    if self.calc_fid is not None:
                        fid = self.calc_fid(fake_samples[-1].squeeze(), images[-1].squeeze(), device=self.device)
                        writer.add_scalar("Frechet Inception Distance", fid, global_step)

                # increment the global_step:
                global_step += 1

            # save to tensorboard
            writer.add_scalars("Loss", {'Generator': mean_generator_loss,
                                        'Discriminator': mean_discriminator_loss}, epoch)
            # if self.calc_fid is not None:
            #     writer.add_scalar("Frechet Inception Distance", mean_fid_score, epoch)

            # store mean_losses into lists
            generator_losses.append(mean_generator_loss)
            discriminator_losses.append(mean_discriminator_loss)

            # calculate the time required for the epoch
            stop_time = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop_time - start_time))

            if checkpoint_factor != 0:
                if epoch % checkpoint_factor == 0:
                    os.makedirs(save_dir, exist_ok=True)

                    model_state_dict = {"epoch": epoch,
                                        "i": global_step,
                                        "gen_state_dict": self.gen.state_dict(),
                                        "disc_state_dict": self.dis.state_dict(),
                                        "gen_optim_state_dict": gen_optim.state_dict(),
                                        "disc_optim_state_dict": dis_optim.state_dict()}

                    th.save(model_state_dict, save_dir+"/model_state_"+str(epoch))

                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(save_dir, "model_ema_state_" + str(epoch) + ".pth")
                        th.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()

        writer.flush()
        writer.close()

        return self
