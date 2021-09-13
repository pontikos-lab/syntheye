import torch
import numpy as np
from helpers.data_utils import get_noise, get_one_hot_labels, combine_vectors


def generate(latent_size,
             n_samples,
             resolution,
             grayscale=True,
             classes=None,
             class_mapping=None,
             generate_randomly=True,
             weights="path/to/weights",
             model_name=None,
             device=None):

    depth = int(np.log2(resolution) - 1)
    mode = "grayscale" if grayscale else "rgb"

    # UNCONDITIONAL GENERATION
    if classes is None and class_mapping is None:

        # ==================================
        # Load model architecture and inputs
        # ==================================

        from models.msggan import msggan
        # load the GAN generator
        gen = msggan.MSG_GAN(depth=depth,
                             latent_size=latent_size,
                             mode=mode,
                             use_ema=True,
                             use_eql=True,
                             ema_decay=0.999,
                             device=device).gen_shadow
        # create the latent vectors
        model_input = get_noise(n_samples, latent_size, device)

        # ====================
        # Load model weights
        # ====================
        state_dict = torch.load(weights)
        gen.load_state_dict(state_dict)
        gen.eval()

        # =====================================
        # Generate image samples
        # =====================================

        generated_images = torch.zeros(n_samples, resolution, resolution)
        for i in range(n_samples):
            generated_images[i, :, :] = gen(torch.unsqueeze(model_input[i, :, :], 0))[-1].squeeze()

        # adjust image pixel values
        from models.msggan import msggan
        generated_images = msggan.Generator.adjust_dynamic_range(generated_images)

    # CONDITIONAL GENERATION
    else:
        n_select_classes = len(classes)
        n_total_classes = len(class_mapping)

        # ==================================
        # Load model architecture and inputs
        # ==================================

        if model_name == "cmsggan":
            from models.msggan import conditional_msggan
            # load the GAN generator
            gen = conditional_msggan.MSG_GAN(depth=depth,
                                             latent_size=latent_size,
                                             n_classes=n_total_classes,
                                             mode=mode,
                                             use_ema=True,
                                             use_eql=True,
                                             ema_decay=0.999,
                                             device=device).gen_shadow
            gen = torch.nn.DataParallel(gen)

            # create latent vector with class embeddings
            noise_input = get_noise(n_samples, latent_size, device) # noise fixed across classes if generate_randomly = False
            class_idxs = torch.tensor([class_mapping[c] for c in classes])
            class_encodings = get_one_hot_labels(class_idxs, n_total_classes).to(device)

            # combine the latent and class vectors together
            model_input = torch.zeros(n_select_classes, n_samples, latent_size + n_total_classes).to(device)
            for i in range(n_select_classes):
                if generate_randomly:
                    # sample new noise every class - noise will not be fixed across class
                    noise_input = get_noise(n_samples, latent_size, device)
                for j in range(n_samples):
                    model_input[i, j, :] = combine_vectors(noise_input[j].view(1, -1),
                                                           class_encodings[i].view(1, -1)).squeeze()

            # ====================
            # Load model weights
            # ====================
            state_dict = torch.load(weights)
            gen.load_state_dict(state_dict)
            gen.eval()

            # =====================================
            # Generate image samples
            # =====================================

            generated_images = torch.zeros(n_select_classes, n_samples, resolution, resolution)
            for i in range(n_select_classes):
                for j in range(n_samples):
                    with torch.no_grad():
                        generated_images[i, j, :, :] = gen(torch.unsqueeze(model_input[i, j, :], 0))[-1].squeeze().to(
                            'cpu')

        elif model_name == "cmsgganv2":
            from models.msggan import conditional_msgganv2
            # load the GAN generator
            gen = conditional_msgganv2.MSG_GAN(depth=depth,
                                               latent_size=latent_size,
                                               n_classes=n_total_classes,
                                               mode=mode,
                                               use_ema=True,
                                               use_eql=True,
                                               ema_decay=0.999,
                                               device=device).gen_shadow
            gen = torch.nn.DataParallel(gen)

            # create latent vector with class embeddings
            noise_input = get_noise(n_samples, latent_size, device)
            class_idxs = torch.tensor([class_mapping[c] for c in classes]).to(device)

            # ====================
            # Load model weights
            # ====================
            state_dict = torch.load(weights)
            gen.load_state_dict(state_dict)
            gen.eval()

            # =====================================
            # Generate image samples
            # =====================================

            generated_images = torch.zeros(n_select_classes, n_samples, resolution, resolution)
            for i in range(n_select_classes):
                if generate_randomly:
                    # generate a new noise every time
                    noise_input = get_noise(n_samples, latent_size, device)
                for j in range(n_samples):
                    with torch.no_grad():
                        z = torch.unsqueeze(noise_input[j], 0)
                        c = torch.tensor([class_idxs[i]]).to(device)
                        generated_images[i, j, :, :] = gen(z, c)[-1].squeeze()

    # adjust image pixel values
    from models.msggan.msggan import Generator
    generated_images = Generator().adjust_dynamic_range(generated_images.detach().to('cpu'))

    return generated_images
