# import libraries
import sys
sys.path.append('..')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import argparse
import copy
import json
from torch.utils.data import DataLoader
from utils.data_utils import ImageDataset
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

# helper functions
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# set global variables
FPATH_COL_NAME = "file.path"
LBL_COL_NAME = "gene"
CLASSES = "../classes.txt"
CLASS_MAPPING = "../classes_mapping.json"
N_CLASSES = 36

def log(message):
    if args.verbose:
        print(message)
    return

def load_data(test_fpath, **kwargs):
    """ Loads transformed data """

    test_transforms = []

    # resizing images
    test_transforms.append(transforms.Resize((299, 299)))

    # convert to grayscale - compulsory
    test_transforms.append(transforms.Grayscale(3))

    # conversion into pytorch tensor - compulsory
    test_transforms.append(transforms.ToTensor())

    # normalize - good for compatibility with inceptionv3
    test_transforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

    # create image transformations list
    test_transforms = transforms.Compose(test_transforms)

    log("Loading Dataset")
    tfold = None #"test" if ("fold" in pd.read_csv(test_fpath).columns) else None
    test_data = ImageDataset(data_file=test_fpath, fpath_col_name=FPATH_COL_NAME, lbl_col_name=LBL_COL_NAME, fold=tfold, class_vals=CLASSES, transforms=test_transforms, class_mapping=CLASS_MAPPING)
    log("{} test images found belonging to {} classes".format(len(test_data), len(test_data.classes)))

    log("Preparing batch loaders from datasets")
    test_data_loader = DataLoader(test_data, 128)
    return test_data_loader

def load_model(config="model_config.json", weights="best_weights.pth", device='cpu'):
    """ Loads a pretrained InceptionV3 model, which is then updated using fine-tuning methods """

    # load model configuration
    model_config = json.load(open(args.config))
    if "model" in model_config.keys():
        name = model_config["model"]
    else:
        name = "inceptionv3"

    # load pretrained InceptionV3 network
    log("Load pretrained {}".format(name))
    if name == "inceptionv3":
        model = torchvision.models.inception_v3(pretrained=True)
        # edit the last layer to have 36 layers only
        model.AuxLogits.fc = nn.Linear(768, N_CLASSES)
        model.fc = nn.Linear(2048, N_CLASSES)
    elif name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, N_CLASSES)
    elif name == "vgg16":
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, N_CLASSES)
    else:
        raise ValueError("Model can only be `inceptionv3`, `vgg16` or `resnet18`")

    if isinstance(device, int):
        model.load_state_dict(torch.load(weights, map_location=torch.device("cuda:"+str(device))))
        return model.to(torch.device("cuda:{}".format(device)))
    elif isinstance(device, list):
        model.load_state_dict(torch.load(weights, map_location=torch.device("cuda:"+str(device[0]))))
        model.to(torch.device("cuda:{}".format(device[0])))
        model = nn.DataParallel(model, device_ids=device)
        return model
    else:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        return model.to("cpu")

def predict(model, dataloader, device='cpu', save_dir=None):

    # create dataframe to store results in
    columns=["file.path", "True Class", "Predicted Class", *("Prob_{}".format(gene) for gene in dataloader.dataset.classes)]
    results_df = pd.DataFrame(columns=columns)

    if isinstance(device, list):
        base_device = torch.device("cuda:{}".format(device[0]))
    elif isinstance(device, int):
        base_device = torch.device("cuda:{}".format(device))
    else:
        base_device = device

    # store dataset sizes
    test_data_size = len(dataloader.dataset)
    
    model.eval()
    running_corrects = 0
    for _, fpath, inputs, labels in tqdm(dataloader):
        inputs = inputs.to(base_device)
        labels = labels.to(base_device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            raw_preds = F.softmax(outputs, dim=1)
            final_preds = torch.argmax(raw_preds, dim=1)

        # statistics per batch
        running_corrects += torch.sum(final_preds == labels.data)

        # add results to dataframe
        row_result = np.concatenate([np.array(fpath)[:, None], labels.cpu().detach().numpy()[:, None], final_preds[:, None].cpu().detach().numpy(), raw_preds.cpu().detach().numpy()], axis=1)
        row_df = pd.DataFrame(row_result, columns=columns)
        results_df = results_df.append(row_df)

    # statistics averaged over entire training set
    acc = running_corrects.double() / test_data_size
    log("Model Accuracy: {:.4f}".format(acc))
    log("Saving Results")
    results_df.to_csv(os.path.join(save_dir, "predictions.csv"))
    return 

if __name__ == "__main__":

    # for parsing command-line arguments
    parser = argparse.ArgumentParser()
    # dataset-related
    parser.add_argument('--config', help="Neural Network model configuration in json file")
    parser.add_argument('--weights', help="Path to weights file for trained model (.pth file)", type=str)
    parser.add_argument('--test', help="Provide a csv file path to the validation set images", type=str)
    parser.add_argument('--resize', default=299, help="Desired dimension to resize image to", type=int)
    parser.add_argument('--normalize', help="Normalizes images with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]", default=1)
    # other training params 
    parser.add_argument('--save-dir', default="experiment_results/", help="Name of directory in which to save test predictions.", type=str)
    parser.add_argument('-v', '--verbose', help="Prints logs of the progress during model training", action="store_true")
    parser.add_argument('-d', '--device', default=0, help="Devices on which model is trained", type=str)
    parser.add_argument('--seed', help="To ensure reproducibility of results", type=int)
    args = parser.parse_args()

    if args.device == "cpu" or not torch.cuda.is_available():
        log("Training on cpu")
        device = args.device
    elif len(args.device) == 1:
        device = int(args.device)
        log("Found 1 GPU : cuda:{}".format(device))
    else:
        device = list(map(int, args.device.split(',')))
        log("Found {} GPUs : ".format(len(device)) + ", ".join(["cuda:{}".format(x) for x in device]))

    # ensures reproducibility
    set_seed(args.seed)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    # create dataloaders
    dataloader = load_data(args.test, **vars(args))

    # load model
    model = load_model(args.config, args.weights, device)

    # begin model inference
    predict(model, dataloader, device=device, save_dir=args.save_dir)