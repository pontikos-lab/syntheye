# import libraries
import os
import concurrent.futures
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from helpers.data_utils import get_one_hot_labels, combine_vectors
from tqdm import tqdm


class Interpolate(nn.Module):
    """
    Interpolates between generated data in terms of classes or latent vectors
    Useful for analysing the latent space behaviour
    """

    def __init__(self, generator, interp, n_classes=10, n_interpolation=9, device=None):
        super(Interpolate, self).__init__()
        self.gen = generator
        self.interp = interp
        self.n_classes = n_classes
        self.n_interp = n_interpolation
        self.device = device

    def interpolate_classes(self, noise, class1, class2):
        first_label = get_one_hot_labels(class1, n_classes=self.n_classes)
        second_label = get_one_hot_labels(class2, n_classes=self.n_classes)

        # calculate percentage of label to incorporate
        percent_second_label = torch.linspace(0, 1, self.n_interp)[:, None]
        interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label

        # combine noise and labels
        noise_and_labels = combine_vectors(noise.repeat(self.n_interp, 1), interpolation_labels.to(self.device))
        fake = self.gen(noise_and_labels)
        return fake

    def interpolate_latents(self, latent1, latent2, label):
        # calculate percentage of label to incorporate
        percent_first_noise = torch.linspace(0, 1, self.n_interp)[:, None].to(self.device)
        interpolation_noise = latent1 * percent_first_noise + latent2 * (1 - percent_first_noise)
        interpolation_label = get_one_hot_labels(label, n_classes=self.n_classes).repeat(self.n_interp, 1).float()

        # combine noise and labels
        noise_and_labels = combine_vectors(interpolation_noise, interpolation_label.to(self.device))
        fake = self.gen(noise_and_labels)
        return fake

    def forward(self, *args):
        if self.interp == "classes":
            fake = self.interpolate_classes(args[0], args[1], args[2])
        elif self.interp == "latents":
            fake = self.interpolate_latents(args[0], args[1], args[2])

        return fake


def compute_fid(gen_imgs, real_images, device=None):
    """ Computes Frechet Inception distance using pytorch's pretrained inceptionv3 model """
    # load inception model
    from torchvision.models import inception_v3
    inception_model = inception_v3(pretrained=True).to(device)
    inception_model = inception_model.eval()  # Evaluation mode
    # use an identity mapping for the final layer instead of a classification layer
    inception_model.fc = nn.Identity()

    # helper functions
    def matrix_sqrt(x):
        y = x.cpu().detach().numpy()
        y = scipy.linalg.sqrtm(y)
        return torch.Tensor(y.real, device=x.device)

    def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
        return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - \
               2*torch.trace(matrix_sqrt(sigma_x @ sigma_y))

    def preprocess(img):
        # img = F.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # img = torchvision.transforms.Normalize(mean, std)(img)
        return img

    def get_covariance(features):
        return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

    # ============================
    # Get the image features
    # ============================
    gen_features_list = []
    real_features_list = []

    # create image dataloaders
    gen_img_dataloader = DataLoader(torch.cat(3 * [torch.unsqueeze(gen_imgs, 1)], dim=1), batch_size=len(gen_imgs), shuffle=False)
    real_img_dataloader = DataLoader(torch.cat(3 * [torch.unsqueeze(real_images, 1)], dim=1), batch_size=len(real_images), shuffle=False)

    for real_example in real_img_dataloader:
        real_example = preprocess(real_example)
        real_features = inception_model(real_example.to(device)).detach().to('cpu')
        real_features_list.append(real_features)

    for gen_example in gen_img_dataloader:
        gen_example = preprocess(gen_example)
        gen_features = inception_model(gen_example.to(device)).detach().to('cpu')
        gen_features_list.append(gen_features)

    # combine all examples into one tensor
    real_features_all = torch.cat(real_features_list)
    gen_features_all = torch.cat(gen_features_list)

    # ============================
    # Calculate feature statistics
    # ============================

    # calculate mean across all observations
    mu_fake = gen_features_all.mean(0)
    mu_real = real_features_all.mean(0)

    # calculate covariance
    sigma_fake = get_covariance(gen_features_all)
    sigma_real = get_covariance(real_features_all)

    with torch.no_grad():
        fid = frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item()

    return fid


def compute_fid_eye2gene(gen_imgs, real_imgs):
    """ Computes frechet inception distance using our Eye2Gene pretrained weights model """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow as tf
    import scipy

    def preprocess(imgs):
        """ Resizes tensors for Eye2gene model """
        imgs = imgs.detach().numpy()
        imgs = imgs[:, :, :, None]
        imgs = np.repeat(imgs, 3, -1)
        return imgs

    # separate preprocess function just for inceptionv3
    inception_preprocess_func = tf.keras.applications.inception_v3.preprocess_input

    # preprocess generated and real images
    gen_imgs, real_imgs = preprocess(gen_imgs), preprocess(real_imgs)
    gen_imgs, real_imgs = inception_preprocess_func(gen_imgs), inception_preprocess_func(real_imgs)

    # load model
    model_paths = os.listdir("models/eye2gene/weights/")
    model_paths = [os.path.join("models/eye2gene/weights", path) for path in model_paths if path.endswith(".h5")]

    inception_model_full = tf.keras.applications.InceptionV3(include_top=True,
                                                             classes=36,
                                                             weights=None,
                                                             input_shape=(256, 256, 3),
                                                             pooling='max')
    inception_model = tf.keras.Model(inputs=inception_model_full.input, outputs=inception_model_full.layers[-2].output)

    # helper functions
    def matrix_sqrt(x):
        y = scipy.linalg.sqrtm(x)
        if np.iscomplexobj(y):
            y = y.real
        return y

    def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
        return (mu_x - mu_y).dot(mu_x - mu_y) + np.trace(sigma_x) + np.trace(sigma_y) - \
               2 * np.trace(matrix_sqrt(sigma_x @ sigma_y))

    def get_covariance(features):
        return np.cov(features, rowvar=False)

    # ============================
    # Get the image features
    # ============================
    gen_features_all = np.zeros((50, 2048))
    real_features_all = np.zeros((50, 2048))
    for path in model_paths:
        inception_model_full.load_weights(path)
        gen_features_all += inception_model.predict(gen_imgs)
        real_features_all += inception_model.predict(real_imgs)

    gen_features_all = gen_features_all / 5
    real_features_all = real_features_all / 5

    # ============================
    # Calculate feature statistics
    # ============================

    # calculate mean across all observations
    mu_fake = gen_features_all.mean(0)
    mu_real = real_features_all.mean(0)

    # calculate covariance
    sigma_fake = get_covariance(gen_features_all)
    sigma_real = get_covariance(real_features_all)

    fid = frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake)

    return fid


def compute_class_confidence(imgs):
    # install libraries
    import tensorflow as tf
    import json
    import matplotlib.pyplot as plt
    import pandas as pd

    # load images
    if isinstance(imgs, (torch.FloatTensor, np.ndarray)):
        images = imgs[:, :, :, None].repeat(3, -1)
    else:
        images = np.zeros((len(imgs), 256, 256, 3))
        for i, img in enumerate(imgs):
            images[i, :, :, :] = plt.imread(img)[:, :, None].repeat(3, -1)

    preprocess_func = tf.keras.applications.inception_v3.preprocess_input
    images = preprocess_func(images)

    # load pretrained eye2gene classifier
    from models.eye2gene.base import Model
    model_paths = os.listdir("models/eye2gene/weights/")
    model_paths = [os.path.join("models/eye2gene/weights", path) for path in model_paths if path.endswith(".h5")]

    conf = np.zeros((50, 36))
    for path in model_paths:
        model = Model().load(path)
        conf += model.predict(images)
    conf = np.divide(conf, len(model_paths))

    # create index to labels converter
    config_path = model_paths[0][:-3] + '.json'
    with open(config_path, 'r') as config_file:
        model_config = json.load(config_file)

    df = pd.DataFrame(conf, columns=model_config['classes'])

    return df


def calc_mutual_information(gen_image, real_image):
    """ Computes the mutual information between a machine-generated image and a real image """
    hist2d, _, _ = np.histogram2d(gen_image.ravel().numpy(), real_image.ravel().numpy())
    pxy = hist2d / float(np.sum(hist2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    MI = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return MI


def mutual_information(gen_imgs, real_dataloader, save_most_similar, save_most_different):
    """ Compares generated images to training dataset images to see how similar they are """

    import time
    from concurrent.futures import ProcessPoolExecutor
    from itertools import product

    start = time.perf_counter()

    # stores all the results for all gen_img/real_img pairs
    all_results = []
    # stores results for gen_img/real_img pairs per iteration of dataloader
    intermediate_results = []

    # stores the five most similar/different image-pairs for each iteration of the dataloader
    most_similar_images = []
    most_different_images = []

    # stores the MI score for the five most similar/different image-pairs for each iteration of the dataloader
    intermediate_similar_results = []
    intermediate_different_results = []

    for real_batch, _ in tqdm(real_dataloader):
        gen_real = list(product(gen_imgs, real_batch))
        with ProcessPoolExecutor() as executor:
            batch_results = [executor.submit(calc_mutual_information, g, r) for g, r in gen_real]
            for r in concurrent.futures.as_completed(batch_results):
                intermediate_results.append(r.result())

        assert len(gen_real) == len(intermediate_results)

        gen_real = [gen_real[i] for i in np.argsort(intermediate_results)]
        intermediate_results = list(np.sort(intermediate_results))

        if save_most_similar:
            most_similar_images += gen_real[-5:]
            intermediate_similar_results += intermediate_results[-5:]
        else:
            most_similar_images = []

        if save_most_different:
            most_different_images += gen_real[:5]
            intermediate_different_results += intermediate_results[:5]
        else:
            most_different_images = []

        all_results += intermediate_results
        intermediate_results = []

    assert len(most_similar_images) == len(intermediate_similar_results)
    assert len(most_different_images) == len(intermediate_different_results)

    if save_most_similar:
        five_most_similar = [most_similar_images[i] for i in np.argsort(intermediate_similar_results)[-5:]]
    else:
        five_most_similar = None

    if save_most_different:
        five_most_different = [most_different_images[i] for i in np.argsort(intermediate_different_results)[:5]]
    else:
        five_most_different = None

    finish = time.perf_counter()
    print("Finished in {} seconds".format(round(finish-start, 2)))

    return all_results, five_most_similar, five_most_different