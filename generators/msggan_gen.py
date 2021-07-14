import torch
import numpy as np
from helpers.data_utils import get_noise, get_one_hot_labels, combine_vectors


def generate(latent_size,
             n_samples,
             resolution,
             grayscale=True,
             classes=None,
             class_mapping=None,
             weights="path/to/weights",
             device=None):

    # ==================================
    # Load model architecture and inputs
    # ==================================
    depth = int(np.log2(resolution) - 1)
    mode = "grayscale" if grayscale else "rgb"

    # only generate random images from task if no conditional information provided
    if classes is None and class_mapping is None:
        from models.msggan import msggan
        # load the GAN model
        gan_model = msggan.MSG_GAN(depth=depth,
                                   latent_size=latent_size,
                                   mode=mode,
                                   use_ema=True,
                                   use_eql=True,
                                   ema_decay=0.999,
                                   device=device)
        gen = gan_model.gen_shadow
        model_input = get_noise(n_samples, latent_size, device)
    # generate images of selected classes
    else:
        from models.msggan import conditional_msggan
        n_select_classes = len(classes)
        n_total_classes = len(class_mapping)
        # load the GAN model
        gan_model = conditional_msggan.MSG_GAN(depth=depth,
                                               latent_size=latent_size,
                                               n_classes=n_total_classes,
                                               mode=mode,
                                               use_ema=True,
                                               use_eql=True,
                                               ema_decay=0.999,
                                               device=device)
        gen = gan_model.gen_shadow
        noise_input = get_noise(n_samples, latent_size, device)
        class_idxs = torch.tensor([class_mapping[i] for i in classes])
        class_encodings = get_one_hot_labels(class_idxs, n_total_classes).to(device)
        model_input = torch.zeros(n_select_classes*n_samples, latent_size+n_total_classes).to(device)

        for i in range(n_select_classes):
            for j in range(n_samples):
                model_input[n_samples*i + j] = combine_vectors(noise_input[j].view(1, -1),
                                                               class_encodings[i].view(1, -1)).squeeze()

    # ====================
    # Load model weights
    # ====================

    state_dict = torch.load(weights)
    gen.load_state_dict(state_dict)
    gen.eval()

    # =======================
    # Generate samples
    # =======================
    generated_images = []
    for i in range(len(model_input)):
        gen_img = gen(torch.unsqueeze(model_input[i], 0))[-1]
        generated_images.append(gen_img.detach().cpu())
    generated_images = torch.cat(generated_images, dim=0)

    # adjust image pixel values
    from models.msggan import msggan
    generated_images = msggan.Generator.adjust_dynamic_range(generated_images)

    return generated_images
