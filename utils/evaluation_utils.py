# import libraries
import concurrent.futures
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
# from scipy.linalg import sqrtm
import cv2
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
from itertools import product


class ComputeSimilarity:
    def __init__(self, metric_name=None):
        self.metric_name = metric_name

    def process_images(self, im, alpha=3.0, beta=0.0, filter="gaussian", fsize=5, threshold="segmentation", tsize=50):

        # basic image processing - changing brightness and contrast
        im = alpha*im + beta
        im = np.clip(im, 0, 255)

        # blur image to get rid of less prominent features
        if filter == "averaging":
            im_filtered = cv2.blur(im, (fsize, fsize))
        elif filter == "gaussian":
            im_filtered = cv2.GaussianBlur(im, (fsize, fsize), 0)
        elif filter == "median":
            im_filtered = cv2.medianBlur(im, 5, (fsize, fsize))
        else:
            im_filtered = im

        # threshold image to segment out prominent features (blood vessels)
        if threshold == "global":
            _, im_filtered = cv2.threshold(im_filtered, tsize, 1, cv2.THRESH_BINARY_INV)
        elif threshold == "adaptive":
            im_filtered = cv2.adaptiveThreshold(im_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        elif threshold == "segmentation":
            from utils.vesselsegmentor.blood_segmentation.generator import RetinalBloodVesselGenerator as segmentor
            seg = segmentor(net_type='unet', pretrained_model='utils/vesselsegmentor/models/unet.pth', using_gpu=False)
            im_filtered = Image.fromarray(im_filtered)
            im_filtered = np.array(seg.generate(im_filtered.convert("RGB")))
        return im_filtered

    def mse(self, im1, im2, in_parallel=True, device='cpu'):
        if in_parallel:
            # standardize the images
            type_ = torch.FloatTensor if device == "cpu" else torch.cuda.FloatTensor
            im1, im2 = im1.type(type_), im2.type(type_)
            im1 = (im1 - torch.mean(im1, dim=(1, 2, 3))[:, None, None, None]) / torch.std(im1, dim=(1, 2, 3))[:, None, None, None]
            im2 = (im2 - torch.mean(im2, dim=(1, 2, 3))[:, None, None, None]) / torch.std(im2, dim=(1, 2, 3))[:, None, None, None]
            # compute mse
            metric_values = torch.sum(im1**2, dim=(1, 2, 3))[:, None] - 2 * torch.einsum('nchw,mchw->nm', im1, im2) + torch.sum(im2 ** 2, dim=(1, 2, 3))[None, :]
            metric_values = metric_values / (im1.shape[-1]*im1.shape[-2])
            metric_values = metric_values.detach().cpu().numpy().ravel()
        else:
            from skimage.metrics import mean_squared_error
            real_synthetic_pairs = list(product(im1, im2))
            with ProcessPoolExecutor() as executor:
                metric_values = [executor.submit(mean_squared_error, s, r) for s, r in real_synthetic_pairs]
                metric_values = [r.result() for r in concurrent.futures.as_completed(metric_values)]
        return metric_values

    def rmse(self, im1, im2, in_parallel):
        if in_parallel:
            metric_values = torch.sum(im2** 2, dim=(1, 2))[:, None] - 2 * torch.einsum('nhw,mhw->nm', im2, im1) + torch.sum(im1 ** 2, dim=(1, 2))[None, :]
            metric_values = torch.sqrt(metric_values.ravel())
        else:
            from skimage.metrics import normalized_root_mse
            real_synthetic_pairs = list(product(im1, im2))
            with ProcessPoolExecutor() as executor:
                metric_values = [executor.submit(normalized_root_mse, s, r) for s, r in real_synthetic_pairs]
                metric_values = [r.result() for r in concurrent.futures.as_completed(metric_values)]
        return metric_values

    def pearsoncorr(self, im1, im2, in_parallel):
        if in_parallel:
            im1, im2 = im1.type(torch.FloatTensor), im2.type(torch.FloatTensor)
            im1_whitened, im2_whitened = im1 - torch.mean(im1, dim=(1, 2))[:, None, None], im2 - torch.mean(im2, dim=(1, 2))[:, None, None]
            num = torch.einsum('mhw,nhw->mn', im1_whitened, im2_whitened)
            denom = torch.sum(im1_whitened**2, dim=(1,2))[:, None] @ torch.sum(im2_whitened**2, dim=(1,2))[:, None].T
            metric_values = num / torch.sqrt(denom)
            metric_values = metric_values.numpy().ravel()
            # raise Exception("Currently, Pearson Correlation cannot be done in parallel! ")
        else:
            from scipy.stats import pearsonr
            real_synthetic_pairs = list(product(im1, im2))
            with ThreadPoolExecutor() as executor:
                metric_values = [executor.submit(pearsonr, s.numpy().ravel(), r.numpy().ravel()) for s, r in real_synthetic_pairs]
                metric_values = [r.result() for r in concurrent.futures.as_completed(metric_values)]
        return metric_values

    def ssim(self, im1, im2, in_parallel):
        if in_parallel:
            raise Exception("Currently SSIM cannot be done in parallel!")
        else:
            from skimage.metrics import structural_similarity
            real_synthetic_pairs = list(product(im1, im2))
            with ProcessPoolExecutor() as executor:
                metric_values = [executor.submit(structural_similarity, s, r) for s, r in real_synthetic_pairs]
                metric_values = [r.result() for r in concurrent.futures.as_completed(metric_values)]
        return metric_values

    def pcacosine(self, im1, im2, pca_model, in_parallel):
        if in_parallel:
            metric_values = pca_model(im1, im2).ravel()
        else:
            from skimage.metrics import structural_similarity
            real_synthetic_pairs = list(product(im1, im2))
            with ProcessPoolExecutor() as executor:
                metric_values = [executor.submit(pca_model, s, r) for s, r in real_synthetic_pairs]
                metric_values = [r.result() for r in concurrent.futures.as_completed(metric_values)]
        return metric_values

    def __call__(self, im1_dataloader, im2_dataloader, process_images=False, return_top=None, **kwargs):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        alpha = kwargs["alpha"] if "alpha" in kwargs.keys() else 3.0
        beta = kwargs["beta"] if "beta" in kwargs.keys() else 0.0
        filter = kwargs["filter"] if "filter" in kwargs.keys() else "gaussian"
        threshold = kwargs["threshold"] if "threshold" in kwargs.keys() else "segmentation"
        fsize = kwargs["fsize"] if "fsize" in kwargs.keys() else 5
        tsize = kwargs["tsize"] if "tsize" in kwargs.keys() else 50

        # select similarity metric function
        if self.metric_name == "MSE":
            func = lambda x, y: self.mse(x, y, in_parallel=True, device=device)
        elif self.metric_name == "PearsonR":
            func = lambda x, y: self.pearsoncorr(x, y, in_parallel=True)
        elif self.metric_name == "RMSE":
            func = lambda x, y: self.rmse(x, y, in_parallel=True)
        elif self.metric_name == "SSIM":
            func = lambda x, y: self.ssim(x, y, in_parallel=False)
        elif self.metric_name == "PCAWithCosine":
            # train PCA model on original dataset
            pca_model = PCAWithCosine(data=im1_dataloader, n_components=1000)
            func = lambda x, y: self.pcacosine(x, y, pca_model=pca_model, in_parallel=True)
        else:
            raise Exception("Metrics can only be MSE/RMSE/SSIM/PCAWithCosine.")

        # store metrics in one dataframe
        metrics = pd.DataFrame(
            columns=["gen_image_index", "real_image_index", "gen_image_path", "real_image_path", self.metric_name])

        for i, spath, simages, _ in tqdm(im1_dataloader):
            for j, rpath, rimages, _ in im2_dataloader:

                # cast tensors to integer type uint8
                simages, rimages = (simages.to(device) * 255).type(torch.ByteTensor), (rimages.to(device) * 255).type(torch.ByteTensor)
                assert len(simages.shape) == 4
                assert len(rimages.shape) == 4

                # compare real and synthetic pairs
                real_synthetic_pair_idxs = np.array(list(product(i, j)))
                real_synthetic_pair_paths = np.array(list(product(spath, rpath)))

                if process_images:
                    process_func = lambda x: self.process_images(x, alpha, beta, filter, fsize, threshold, tsize)
                    # process real and synthetic images
                    simages, rimages = np.uint8(simages.detach().cpu().numpy()), np.uint8(rimages.detach().cpu().numpy())
                    simages = np.array(list(map(process_func, simages)))
                    rimages = np.array(list(map(process_func, rimages)))
                    # simages = [self.process_images(simages[i], alpha, beta, filter, fsize, threshold, tsize) for i in range(len(simages))]
                    # rimages = [self.process_images(rimages[i], alpha, beta, filter, fsize, threshold, tsize) for i in range(len(rimages))]
                    type_ = torch.ByteTensor if device == "cpu" else torch.cuda.ByteTensor 
                    simages, rimages = torch.as_tensor(simages).type(type_), torch.as_tensor(rimages).type(type_)

                # compute similarity metric given two tensors of dimension (M, X, X) and (N, X, X)
                metric_values = func(simages, rimages)

                assert len(metric_values) == len(real_synthetic_pair_paths), \
                    "Number of metric values computed = {}, Number of pairs found = {}".format(len(metric_values),
                                                                                               len(real_synthetic_pair_paths))

                # save results to dataframe
                df = pd.DataFrame(np.concatenate((np.array(real_synthetic_pair_idxs),
                                                  np.array(real_synthetic_pair_paths),
                                                  np.array(metric_values)[:, None]), axis=1),
                                  columns=["gen_image_index", "real_image_index", "gen_image_path", "real_image_path",
                                           self.metric_name])

                # update dataframe
                metrics = pd.concat([metrics, df])

        ascending = True if self.metric_name in ["MSE", "RMSE", "PSNR"] else False
        metrics = metrics.sort_values(by=self.metric_name, ascending=ascending, ignore_index=True)
        if return_top is not None:
            return metrics.head(return_top)
        else:
            return metrics


class ComputeQuality:
    def __init__(self, quality_metric=None):
        self.quality_metric = quality_metric

    def __call__(self, synthetic_dataloader):
        if self.quality_metric == "BRISQUE":
            # import imquality.brisque as brisque
            from brisque import BRISQUE
            func = BRISQUE().get_score  # brisque.score
        else:
            raise Exception("Only options are BRISQUE!")

        all_quality_vals = pd.DataFrame(columns=["Synthetic image path", "Quality Score"])
        for _, im_paths, im_batch, _ in synthetic_dataloader:
            from brisque import BRISQUE
            brisq = BRISQUE()
            im_batch = (im_batch * 255).type(torch.ByteTensor)
            # Image.fromarray(im_batch[i].numpy().squeeze())
            quality_vals = [brisq.get_score(np.array(Image.open(im_paths[i]))) for i in range(len(im_batch))]
            quality_vals = list(zip(im_paths, quality_vals))
            df = pd.DataFrame(np.array(quality_vals), columns=["Synthetic image path", "Quality Score"])
            all_quality_vals = pd.concat([all_quality_vals, df])

        return all_quality_vals


class Interpolate(nn.Module):
    """
    Interpolates between generated data in terms of classes or latent vectors
    Useful for analysing the latent space behaviour
    """

    def __init__(self, generator, interp, n_classes=10, n_interpolation=9, component=None, device=None):
        super(Interpolate, self).__init__()
        self.gen = generator
        self.interp = interp
        self.component = component
        self.n_classes = n_classes
        self.n_interp = n_interpolation
        self.device = device

    def linear_interp(self, vec1, vec2):
        percent_second = torch.linspace(0, 1, self.n_interp)[:, None].to(self.device)
        interpolation = vec1*(1-percent_second) + vec2*percent_second
        return interpolation

    def slerp(self, vec1, vec2):
        percent_second = torch.linspace(0, 1, self.n_interp)[:, None].to(self.device)
        omega = torch.arccos(torch.clip(torch.matmul(vec1/torch.norm(vec1), (vec2/torch.norm(vec2)).T), -1, 1))
        so = torch.sin(omega)
        interpolation = (torch.sin((1.0 - percent_second)*omega.item())/so.item()) * vec1 + (torch.sin(percent_second*omega.item()) / so.item()) * vec2
        return interpolation

    def interpolate_classes(self, noise, class1, class2):
        first_label = get_one_hot_labels(class1, n_classes=self.n_classes).type(torch.FloatTensor).to(self.device)
        second_label = get_one_hot_labels(class2, n_classes=self.n_classes).type(torch.FloatTensor).to(self.device)

        assert first_label.shape == (1, self.n_classes), "{}".format(first_label.shape)
        assert second_label.shape == (1, self.n_classes), "{}".format(second_label.shape)

        if self.interp == "linear":
            interpolation_labels = self.linear_interp(first_label, second_label)
        elif self.interp == "slerp":
            interpolation_labels = self.slerp(first_label, second_label)
        else:
            raise Exception("Can only interpolate linearly or using slerp.")

        # combine noise and labels
        noise_and_labels = combine_vectors(noise.repeat(self.n_interp, 1), interpolation_labels.to(self.device))
        fake = self.gen(noise_and_labels)
        return fake

    def interpolate_noise(self, latent1, latent2, label):
        # calculate percentage of label to incorporate
        if self.interp == "linear":
            interpolation_noise = self.linear_interp(latent1, latent2)
        elif self.interp == "slerp":
            interpolation_noise = self.slerp(latent1, latent2)
        else:
            raise Exception("Can only interpolate linearly or using slerp.")

        interpolation_label = get_one_hot_labels(label, n_classes=self.n_classes).repeat(self.n_interp, 1).float()

        # combine noise and labels
        noise_and_labels = combine_vectors(interpolation_noise, interpolation_label.to(self.device))
        fake = self.gen(noise_and_labels)
        return fake

    def forward(self, *args):
        if self.component == "classes":
            fake = self.interpolate_classes(args[0], args[1], args[2])
        elif self.component == "latents":
            fake = self.interpolate_noise(args[0], args[1], args[2])
        else:
            raise Exception("Can only interpolate between classes or noise vectors")
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


def compute_fid_parallel(gen_imgs, real_imgs, mode="rgb", device=None):
    """ Computes Frechet Inception distance using pytorch's pretrained inceptionv3 model """
    # load inception model
    from torchvision.models import inception_v3
    from torch.nn import DataParallel
    inception_model = inception_v3(pretrained=True).to(device)
    inception_model = inception_model.eval()  # Evaluation mode
    # use an identity mapping for the final layer instead of a classification layer
    inception_model.fc = nn.Identity()
    # inception_model = DataParallel(inception_model, device_ids=list(range(torch.cuda.device_count())))

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
    if mode == "grayscale":
        real_example = preprocess(torch.cat(3 * [real_imgs], dim=1))
        gen_example = preprocess(torch.cat(3 * [gen_imgs], dim=1))
    else:
        real_example = preprocess(real_imgs)
        gen_example = preprocess(gen_imgs)

    real_features_list = inception_model(real_example.to(device))
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


# def calc_l2_norm(gen_imgs, real_imgs):
#     """ Computes Euclidean Distance between generated and real images """
#     result = torch.sum(gen_imgs**2, dim=(1, 2))[:, None] - 2*torch.einsum('nhw,mhw->nm', gen_imgs, real_imgs) + torch.sum(real_imgs**2, dim=(1,2))[None, :]
#     return result


# def calc_pearson_corr(gen_image, real_image):
#     """ Computes Pearson Correlation Coefficient between generated and real images """
#     from scipy.stats import pearsonr
#     return pearsonr(gen_image.ravel(), real_image.ravel())[0]


# def calc_img_similarity(gen_imgs, real_dataloader, similarity_metric, save_most_similar, save_most_different):
#     """ Compares generated images to training dataset images to see how similar they are """
#
#     import time
#     from concurrent.futures import ProcessPoolExecutor
#     from itertools import product
#     from sewar.full_ref import mse, rmse, ssim, scc, psnr
#     from skimage.metrics import structural_similarity, mean_squared_error
#
#     # select similarity metric
#     if similarity_metric == "MSE":
#         metric = mean_squared_error
#     elif similarity_metric == "RMSE":
#         metric = rmse
#     elif similarity_metric == "SSIM":
#         metric = structural_similarity #StructuralSimilarity(device="cuda")
#     elif similarity_metric == "corr":
#         metric = calc_pearson_corr
#     elif similarity_metric == "SCC":
#         metric = scc
#     elif similarity_metric == "PSNR":
#         metric = psnr
#     else:
#         metric = None
#
#     # stores all the results for all gen_img/real_img pairs
#     all_results = []
#
#     # stores the five most similar/different image-pairs for each iteration of the dataloader
#     most_similar_images = []
#     most_different_images = []
#
#     # stores the MI score for the five most similar/different image-pairs for each iteration of the dataloader
#     intermediate_similar_results = []
#     intermediate_different_results = []
#
#     print("Comparing original image with itself...")
#     if similarity_metric == "SSIM":
#         # calibrate = [metric(g.to("cuda")[None, None, :, :]*255, g.to("cuda")[None, None, :, :]*255) for g in gen_imgs]
#         calibrate = [metric(np.uint8(g.numpy()*255), np.uint8(g.numpy()*255)) for g in gen_imgs]
#     else:
#         calibrate = [metric(np.uint8(g.numpy()*255), np.uint8(g.numpy()*255)) for g in gen_imgs]
#     print(calibrate)
#
#     for real_batch, _ in tqdm(real_dataloader):
#         # compute the similarity between generated images and batch of real images
#         gen_real = list(product(gen_imgs, real_batch.squeeze()))
#         if similarity_metric == "SSIM":
#             # intermediate_results = [metric(g.to("cuda")[None, None, :, :]*255, r.to("cuda")[None, None, :, :]*255) for g, r in gen_real]
#             intermediate_results = [metric(np.uint8(g.numpy() * 255), np.uint8(r.numpy() * 255)) for g, r in gen_real]
#         else:
#             intermediate_results = [metric(np.uint8(g.numpy() * 255), np.uint8(r.numpy() * 255)) for g, r in gen_real]
#
#         assert len(gen_real) == len(intermediate_results)
#
#         # sort the pairwise results based on similarity metric value
#         gen_real = [gen_real[i] for i in np.argsort(intermediate_results)]
#         intermediate_results = list(np.sort(intermediate_results))
#
#         # save the most similar from this batch of pairwise comparisons
#         if save_most_similar:
#             if similarity_metric in ["MSE", "RMSE"]: # if rmse or mse, then the smallest value is greatest similarity
#                 most_similar_images += gen_real[:5]
#                 intermediate_similar_results += intermediate_results[:5]
#             else:
#                 most_similar_images += gen_real[-5:]
#                 intermediate_similar_results += intermediate_results[-5:]
#         else:
#             most_similar_images = []
#
#         if save_most_different:
#             if similarity_metric in ["MSE", "RMSE"]:
#                 most_different_images += gen_real[-5:]
#                 intermediate_different_results += intermediate_results[-5:]
#             else:
#                 most_different_images += gen_real[:5]
#                 intermediate_different_results += intermediate_results[:5]
#         else:
#             most_different_images = []
#
#         # save all the metrics between gen and real batch
#         all_results += intermediate_results
#
#     assert len(most_similar_images) == len(intermediate_similar_results)
#     assert len(most_different_images) == len(intermediate_different_results)
#
#     if save_most_similar:
#         if similarity_metric in ["MSE", "RMSE"]:
#             five_most_similar = [most_similar_images[i] for i in np.argsort(intermediate_similar_results)[:5]]
#         else:
#             five_most_similar = [most_similar_images[i] for i in np.argsort(intermediate_similar_results)[-5:]]
#     else:
#         five_most_similar = None
#
#     if save_most_different:
#         if similarity_metric in ["MSE", "RMSE"]:
#             five_most_different = [most_different_images[i] for i in np.argsort(intermediate_different_results)[-5:]]
#         else:
#             five_most_different = [most_different_images[i] for i in np.argsort(intermediate_different_results)[:5]]
#     else:
#         five_most_different = None
#
#     return all_results, five_most_similar, five_most_different
#
#
# def calc_img_similarity_v2(synthetic_dataloader,
#                            real_dataloader,
#                            filter=None,
#                            threshold=None,
#                            similarity_metric="MSE",
#                            return_top=None):
#
#     """ Compares generated images to training dataset images to see how similar they are """
#
#     from itertools import product
#
#     def process_image(image):
#
#         # basic image processing
#         alpha = 3.0
#         beta = 1.0
#         image = alpha*image + beta
#         image = np.clip(image, 0, 255)
#
#         if filter == "averaging":
#             im_filtered = cv2.blur(image, (7, 7))
#         elif filter == "gaussian":
#             im_filtered = cv2.GaussianBlur(image, (7, 7), 0)
#         elif filter == "median":
#             im_filtered = cv2.medianBlur(image, 5, (7, 7))
#         else:
#             im_filtered = image
#
#         if threshold == "global":
#             _, im_filtered = cv2.threshold(im_filtered, 50, 1, cv2.THRESH_BINARY_INV)
#         elif threshold == "adaptive":
#             im_filtered = cv2.adaptiveThreshold(im_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#         elif threshold == "segment":
#             from utils.vesselsegmentor.blood_segmentation.generator import RetinalBloodVesselGenerator as segmentor
#             seg = segmentor(net_type='unet', pretrained_model='utils/vesselsegmentor/models/unet.pth', using_gpu=False)
#             im_filtered = Image.fromarray(im_filtered)
#             im_filtered = np.array(seg.generate(im_filtered.convert("RGB")))
#         return im_filtered
#
#     def metric(image_pairs, **kwargs):
#         image1, image2 = image_pairs
#         from sewar.full_ref import mse, rmse, ssim, scc, psnr
#         from skimage.metrics import structural_similarity, mean_squared_error
#         if similarity_metric == "MSE":
#             sim = mean_squared_error(image1, image2)
#         elif similarity_metric == "RMSE":
#             sim = rmse(image1, image2)
#         elif similarity_metric == "SSIM":
#             sim = ssim(image1, image2)
#         elif similarity_metric == "PearsonCorr":
#             sim = calc_pearson_corr(image1, image2)
#         elif similarity_metric == "PSNR":
#             sim = psnr(image1, image2)
#         elif similarity_metric == "PCAWithCosine":
#             model = PCAWithCosine(data=real_dataloader, n_components=1000)
#             sim = model(image1, image2, return_diagonal=kwargs["return_diagonal"])
#         else:
#             raise Exception("Incorrect metric specified!")
#         return sim
#
#     # additional parameters to select most similar/different results
#     ascending = True if similarity_metric in ["MSE", "RMSE", "PSNR"] else False
#     in_parallel = True if similarity_metric in ["PCAWithCosine"] else False
#
#     print("Comparing real images versus themselves...")
#     _, _, real_sample, _ = next(iter(real_dataloader))
#     real_sample = (real_sample[:10]*255).numpy()
#     real_vs_real = metric((real_sample, real_sample), return_diagonal=True)
#     print(real_vs_real)
#
#     print("Comparing generated images versus themselves...")
#     _, _, gen_sample, _ = next(iter(synthetic_dataloader))
#     gen_sample = (gen_sample[:10] * 255).numpy()
#     gen_vs_gen = metric((gen_sample, gen_sample), return_diagonal=True)
#     print(gen_vs_gen)
#
#     # store metrics in one dataframe
#     metrics = pd.DataFrame(columns=["gen_image_index", "real_image_index", "gen_image_path", "real_image_path", similarity_metric])
#
#     for i, spath, simages, stargets in synthetic_dataloader:
#         for j, rpath, rimages, rtargets in tqdm(real_dataloader):
#
#             # compare real and synthetic pairs
#             real_synthetic_pair_idxs = np.array(list(product(i, j)))
#             real_synthetic_pair_paths = np.array(list(product(spath, rpath)))
#
#             # process real and synthetic images
#             simages, rimages = np.uint8(simages.squeeze().numpy() * 255), np.uint8(rimages.squeeze().numpy() * 255)
#
#             # simages = [process_image(simages[i]) for i in range(simages.shape[0])]
#             # rimages = [process_image(rimages[i]) for i in range(rimages.shape[0])]
#
#             # compute metric in parallel or sequentially
#             if in_parallel:
#                 metric_values = metric(simages, rimages).ravel()
#             else:
#                 real_synthetic_pairs = list(product(simages, rimages))
#                 with ProcessPoolExecutor() as executor:
#                     metric_values = executor.map(metric, real_synthetic_pairs)
#                     # metric_values = [executor.submit(metric, s, r) for s, r in real_synthetic_pairs]
#                     metric_values = [r.result() for r in concurrent.futures.as_completed(metric_values)]
#
#             assert len(metric_values) == len(real_synthetic_pair_paths),\
#                 "Number of metric values computed = {}, Number of pairs found = {}".format(len(metric_values),
#                                                                                            len(real_synthetic_pair_paths))
#
#             # save results to dataframe
#             df = pd.DataFrame(np.concatenate((np.array(real_synthetic_pair_idxs), np.array(real_synthetic_pair_paths), np.array(metric_values)[:, None]), axis=1),
#                               columns=["gen_image_index", "real_image_index", "gen_image_path", "real_image_path", similarity_metric])
#
#             # update dataframe
#             metrics = metrics.append(df, ignore_index=True)
#
#     metrics = metrics.sort_values(by=similarity_metric, ascending=ascending, ignore_index=True)
#     if return_top is not None:
#         return metrics.head(return_top)
#     else:
#         return metrics

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