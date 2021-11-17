# import libraries
import concurrent.futures
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
# from scipy.linalg import sqrtm
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.data_utils import get_one_hot_labels, combine_vectors
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score


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


class StructuralSimilarity(nn.Module):
    """ Computes structural similarity metric between images"""
    def __init__(self, channel=1, val_range=255, window_size=11, sigma=1.5, device="cpu"):
        super(StructuralSimilarity, self).__init__()
        self.device = device
        self.L = val_range
        self.window = self.create_gaussian_kernel(window_size, sigma, channel).to(self.device)
        self.kernel = lambda x: F.conv2d(x, self.window, padding=window_size//2, groups=channel)

    def create_gaussian_kernel(self, window_size, sigma, channel=1):
        import math
        # create 1d kernel
        kernel1d = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        # create 2d kernel
        kernel2d = kernel1d[:, None] @ kernel1d[:, None].T
        kernel2d = torch.Tensor(kernel2d.expand(channel, 1, window_size, window_size).contiguous())

        assert kernel2d.shape == (channel, 1, window_size, window_size)
        return kernel2d

    def forward(self, im1, im2):

        assert im1.shape == im2.shape, "Images should be same size!"

        im1 = im1.to(self.device)
        im2 = im2.to(self.device)

        # compute luminescence
        mu1 = self.kernel(im1)
        mu2 = self.kernel(im2)
        mu12 = mu1*mu2
        mu1_sq = mu1**2
        mu2_sq = mu2**2

        # compute contrast metric
        sigma1_sq = self.kernel(im1*im1) - mu1_sq
        sigma2_sq = self.kernel(im2*im2) - mu2_sq
        sigma12 = self.kernel(im1*im2) - mu12

        # stability constants
        C1 = (0.01)**2
        C2 = (0.03)**2

        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)

        ssim = ((2 * mu12 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim.mean().item()


class PCAWithCosine(nn.Module):
    def __init__(self, data, n_components):
        super(PCAWithCosine, self).__init__()
        self.data = data
        self.n_components = n_components
        self.pca_transformer = self.pca_fit_data()

    def pca_fit_data(self):
        from sklearn.decomposition import IncrementalPCA
        transformer = IncrementalPCA(n_components=self.n_components, batch_size=self.data.batch_size)
        print("Fitting PCA to real dataset...")
        for i, _, x, _ in tqdm(self.data):
            batch = x.view(x.shape[0], -1).numpy()
            assert batch.shape == (x.shape[0], x.shape[-1]*x.shape[-2]),\
                "Your Batch shape: {}, Expected batch shape: {}".format(batch.shape, (x.shape[0], x.shape[-1] * x.shape[-2]))
            transformer.partial_fit(batch)
        return transformer

    def forward(self, im1, im2, return_diagonal=False):
        im1 = im1.squeeze()
        im2 = im2.squeeze()
        if len(im1.shape) == 2:
            im1_dimreduce = self.pca_transformer.transform(im1.reshape(1, -1))
            im2_dimreduce = self.pca_transformer.transform(im2.reshape(1, -1))
            sim = (im1_dimreduce @ im2_dimreduce.T) / (np.linalg.norm(im1_dimreduce) * np.linalg.norm(im2_dimreduce))
            sim = sim.item()
        else:
            im1_dimreduce = self.pca_transformer.transform(im1.reshape(len(im1), -1))
            im2_dimreduce = self.pca_transformer.transform(im2.reshape(len(im2), -1))
            # compute cosine similarity in parallel
            sim = np.matmul(im1_dimreduce, im2_dimreduce.T) / np.outer(np.linalg.norm(im1_dimreduce, axis=1), np.linalg.norm(im2_dimreduce, axis=1))
            if return_diagonal:
                sim = np.diagonal(sim)
            else:
                sim = sim.ravel()
        return sim


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
        img = F.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = transforms.Normalize(mean, std)(img)
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


def compute_fid_parallel(gen_imgs, real_imgs, device=None):
    """ Computes Frechet Inception distance using pytorch's pretrained inceptionv3 model """
    # load inception model
    from torchvision.models import inception_v3
    inception_model = inception_v3(pretrained=True).to(device)
    inception_model = inception_model.eval()  # Evaluation mode
    # use an identity mapping for the final layer instead of a classification layer
    inception_model.fc = nn.Identity()

    # helper functions
    def matrix_sqrt(x):
        y = x.to('cpu').detach().numpy()
        y = scipy.linalg.sqrtm(y)
        return torch.Tensor(y.real).to(x.device)

    def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
        return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - \
               2*torch.trace(matrix_sqrt(sigma_x @ sigma_y))

    def preprocess(img):
        img = F.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = transforms.Normalize(mean, std)(img)
        return img

    def get_covariance(features):
        return torch.cov(features.T)

    # ============================
    # Get the image features
    # ============================
    real_example = preprocess(torch.cat(3 * [real_imgs], dim=1))
    real_features_list = inception_model(real_example.to(device))

    gen_example = preprocess(torch.cat(3 * [gen_imgs], dim=1))
    gen_features_list = inception_model(gen_example.to(device))

    # ============================
    # Calculate feature statistics
    # ============================

    mu_fake = torch.mean(gen_features_list, dim=0)
    mu_real = torch.mean(real_features_list, dim=0)

    # calculate covariance
    sigma_fake = get_covariance(gen_features_list)
    sigma_real = get_covariance(real_features_list)

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
    """ Computes the mutual information between a generated image and a real image """
    hist2d, _, _ = np.histogram2d(gen_image.numpy().ravel(), real_image.numpy().ravel(), bins=20)
    pxy = hist2d / float(np.sum(hist2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    MI = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    nxs = px > 0
    nys = py > 0
    MI = 2 * MI / (-1 * np.sum(px[nxs] * np.log(px[nxs])) + -1 * np.sum(py[nys] * np.log(py[nys])))
    return MI


def calc_l2_norm(gen_imgs, real_imgs):
    """ Computes Euclidean Distance between generated and real images """
    result = torch.sum(gen_imgs**2, dim=(1, 2))[:, None] - 2*torch.einsum('nhw,mhw->nm', gen_imgs, real_imgs) + torch.sum(real_imgs**2, dim=(1,2))[None, :]
    return result


def calc_pearson_corr(gen_image, real_image):
    """ Computes Pearson Correlation Coefficient between generated and real images """
    from scipy.stats import pearsonr
    return pearsonr(gen_image.ravel(), real_image.ravel())[0]


def calc_img_similarity(gen_imgs, real_dataloader, similarity_metric, save_most_similar, save_most_different):
    """ Compares generated images to training dataset images to see how similar they are """

    import time
    from concurrent.futures import ProcessPoolExecutor
    from itertools import product
    from sewar.full_ref import mse, rmse, ssim, scc, psnr
    from skimage.metrics import structural_similarity, mean_squared_error

    # select similarity metric
    if similarity_metric == "MSE":
        metric = mean_squared_error
    elif similarity_metric == "RMSE":
        metric = rmse
    elif similarity_metric == "SSIM":
        metric = structural_similarity #StructuralSimilarity(device="cuda")
    elif similarity_metric == "corr":
        metric = calc_pearson_corr
    elif similarity_metric == "SCC":
        metric = scc
    elif similarity_metric == "PSNR":
        metric = psnr
    else:
        metric = None

    # stores all the results for all gen_img/real_img pairs
    all_results = []

    # stores the five most similar/different image-pairs for each iteration of the dataloader
    most_similar_images = []
    most_different_images = []

    # stores the MI score for the five most similar/different image-pairs for each iteration of the dataloader
    intermediate_similar_results = []
    intermediate_different_results = []

    print("Comparing original image with itself...")
    if similarity_metric == "SSIM":
        # calibrate = [metric(g.to("cuda")[None, None, :, :]*255, g.to("cuda")[None, None, :, :]*255) for g in gen_imgs]
        calibrate = [metric(np.uint8(g.numpy()*255), np.uint8(g.numpy()*255)) for g in gen_imgs]
    else:
        calibrate = [metric(np.uint8(g.numpy()*255), np.uint8(g.numpy()*255)) for g in gen_imgs]
    print(calibrate)

    for real_batch, _ in tqdm(real_dataloader):
        # compute the similarity between generated images and batch of real images
        gen_real = list(product(gen_imgs, real_batch.squeeze()))
        if similarity_metric == "SSIM":
            # intermediate_results = [metric(g.to("cuda")[None, None, :, :]*255, r.to("cuda")[None, None, :, :]*255) for g, r in gen_real]
            intermediate_results = [metric(np.uint8(g.numpy() * 255), np.uint8(r.numpy() * 255)) for g, r in gen_real]
        else:
            intermediate_results = [metric(np.uint8(g.numpy() * 255), np.uint8(r.numpy() * 255)) for g, r in gen_real]

        assert len(gen_real) == len(intermediate_results)

        # sort the pairwise results based on similarity metric value
        gen_real = [gen_real[i] for i in np.argsort(intermediate_results)]
        intermediate_results = list(np.sort(intermediate_results))

        # save the most similar from this batch of pairwise comparisons
        if save_most_similar:
            if similarity_metric in ["MSE", "RMSE"]: # if rmse or mse, then the smallest value is greatest similarity
                most_similar_images += gen_real[:5]
                intermediate_similar_results += intermediate_results[:5]
            else:
                most_similar_images += gen_real[-5:]
                intermediate_similar_results += intermediate_results[-5:]
        else:
            most_similar_images = []

        if save_most_different:
            if similarity_metric in ["MSE", "RMSE"]:
                most_different_images += gen_real[-5:]
                intermediate_different_results += intermediate_results[-5:]
            else:
                most_different_images += gen_real[:5]
                intermediate_different_results += intermediate_results[:5]
        else:
            most_different_images = []

        # save all the metrics between gen and real batch
        all_results += intermediate_results

    assert len(most_similar_images) == len(intermediate_similar_results)
    assert len(most_different_images) == len(intermediate_different_results)

    if save_most_similar:
        if similarity_metric in ["MSE", "RMSE"]:
            five_most_similar = [most_similar_images[i] for i in np.argsort(intermediate_similar_results)[:5]]
        else:
            five_most_similar = [most_similar_images[i] for i in np.argsort(intermediate_similar_results)[-5:]]
    else:
        five_most_similar = None

    if save_most_different:
        if similarity_metric in ["MSE", "RMSE"]:
            five_most_different = [most_different_images[i] for i in np.argsort(intermediate_different_results)[-5:]]
        else:
            five_most_different = [most_different_images[i] for i in np.argsort(intermediate_different_results)[:5]]
    else:
        five_most_different = None

    return all_results, five_most_similar, five_most_different


def calc_img_similarity_v2(synthetic_dataloader,
                           real_dataloader,
                           similarity_metric,
                           return_top=None):

    """ Compares generated images to training dataset images to see how similar they are """

    from itertools import product
    from sewar.full_ref import mse, rmse, ssim, scc, psnr
    from skimage.metrics import structural_similarity, mean_squared_error

    # select similarity metric
    if similarity_metric == "MSE":
        metric = mean_squared_error
        in_parallel = False
        ascending = True
    elif similarity_metric == "RMSE":
        metric = rmse
        in_parallel = False
        ascending = True
    elif similarity_metric == "SSIM":
        metric = structural_similarity
        in_parallel = False
        ascending = False
    elif similarity_metric == "corr":
        metric = calc_pearson_corr
        in_parallel = False
        ascending = False
    elif similarity_metric == "SCC":
        metric = scc
        in_parallel = False
        ascending = False
    elif similarity_metric == "PSNR":
        metric = psnr
        in_parallel = False
        ascending = True
    elif similarity_metric == "PCAWithCosine":
        metric = PCAWithCosine(data=real_dataloader, n_components=1000)
        # print(np.cumsum(metric.pca_transformer.explained_variance_ratio_))
        in_parallel = True
        ascending = False
    else:
        metric = None
        in_parallel = None
        ascending = None

    print("Comparing real images versus themselves...")
    _, _, real_sample, _ = next(iter(real_dataloader))
    real_vs_real = metric((real_sample[:10] * 255).numpy(), (real_sample[:10] * 255).numpy(), return_diagonal=True)
    print(real_vs_real)

    print("Comparing generated images versus themselves...")
    _, _, gen_sample, _ = next(iter(synthetic_dataloader))
    gen_vs_gen = metric((gen_sample[:10] * 255).numpy(), (gen_sample[:10] * 255).numpy(), return_diagonal=True)
    print(gen_vs_gen)

    # mask the images
    mask = Image.open("mask.png")
    mask = transforms.ToTensor()(transforms.Grayscale()(transforms.Resize((256, 256))(mask)))

    # store metrics in one dataframe
    metrics = pd.DataFrame(columns=["gen_image_index", "real_image_index", "gen_image_path", "real_image_path", similarity_metric])

    for i, spath, simages, stargets in synthetic_dataloader:
        for j, rpath, rimages, rtargets in tqdm(real_dataloader):

            # compare real and synthetic pairs
            real_synthetic_pair_idxs = np.array(list(product(i, j)))
            real_synthetic_pair_paths = np.array(list(product(spath, rpath)))

            # compute metric in parallel or sequentially
            if in_parallel:
                metric_values = metric(simages, rimages).ravel()
            else:
                real_synthetic_pairs = list(product(simages, rimages))
                with ProcessPoolExecutor() as executor:
                    metric_values = [executor.submit(metric, np.uint8((s*mask).numpy()*255), np.uint8((r*mask).numpy()*255)) for s, r in real_synthetic_pairs]
                    metric_values = [r.result() for r in concurrent.futures.as_completed(metric_values)]

            assert len(metric_values) == len(real_synthetic_pair_paths), "Number of metric values computed = {}, Number of pairs found = {}".format(len(metric_values), len(real_synthetic_pair_paths))

            # save results to dataframe
            df = pd.DataFrame(np.concatenate((np.array(real_synthetic_pair_idxs), np.array(real_synthetic_pair_paths), np.array(metric_values)[:, None]), axis=1),
                              columns=["gen_image_index", "real_image_index", "gen_image_path", "real_image_path", similarity_metric])

            # update dataframe
            metrics = metrics.append(df, ignore_index=True)

    metrics = metrics.sort_values(by=similarity_metric, ascending=ascending, ignore_index=True)
    if return_top is not None:
        return metrics.head(return_top)
    else:
        return metrics

# def calc_img_similarity(gen_imgs, real_dataloader, save_most_similar, save_most_different):
#     """ Compares generated images to training dataset images to see how similar they are """
#
#     import time
#     start = time.perf_counter()
#
#     five_most_similar = np.zeros((len(gen_imgs), ))
#
#     for g in gen_imgs:
#         for real_batch, _ in tqdm(real_dataloader):
#             intermediate_results = [calc_pearson_corr(g, r) for r in real_batch]
#
#             gen_real = [gen_real[i] for i in np.argsort(intermediate_results)]
#             intermediate_results = list(np.sort(intermediate_results))
#
#             all_results += intermediate_results
#             intermediate_results = []
#
#     if save_most_similar:
#         five_most_similar = [most_similar_images[i] for i in np.argsort(intermediate_similar_results)[-5:]]
#     else:
#         five_most_similar = None
#
#     if save_most_different:
#         five_most_different = [most_different_images[i] for i in np.argsort(intermediate_different_results)[:5]]
#     else:
#         five_most_different = None
#
#     finish = time.perf_counter()
#     print("Finished in {} seconds".format(round(finish-start, 2)))
#
#     return all_results, five_most_similar, five_most_different

# with ProcessPoolExecutor() as executor:
#     batch_results = [executor.submit(calc_pearson_corr, g, r) for g, r in gen_real]
#     # batch_results = [executor.submit(calc_l2_norm, gen_imgs, real_batch.squeeze())]
#     for r in concurrent.futures.as_completed(batch_results):
#         intermediate_results.append(r.result())
#         # intermediate_results += list(r.result().reshape(gen_imgs.shape[0]*real_batch.shape[0]))


        # real_vs_real_v2 = []
        # for i in range(10):
        #     for j in range(10):
        #         real_vs_real_v2.append(metric(np.uint8(real_sample[i].numpy()*255), np.uint8(real_sample[j].numpy()*255)))
        # print(real_vs_real)
        # print(real_vs_real_v2)

        # real_sample = np.uint8((real_sample*255).squeeze().numpy().reshape(len(real_sample), -1))
        # real_sample_features = metric.pca_transformer.transform(real_sample)
        # real_vs_real = (real_sample_features @ real_sample_features.T) / np.outer(np.linalg.norm(real_sample_features, axis=1), np.linalg.norm(real_sample_features, axis=1))