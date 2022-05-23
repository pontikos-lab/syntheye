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
import seaborn as sns

# helper functions
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# set global variables
FPATH_COL_NAME = "file.path"
LBL_COL_NAME = "gene"
CLASSES = "../classes.txt"
with open(CLASSES) as f:
    CLASS_LIST = f.read().splitlines()
CLASS_MAPPING = "../classes_mapping.json"
CLASS2IDX = json.load(open(CLASS_MAPPING))
IDX2CLASS = {v:k for k, v in CLASS2IDX.items()}
N_CLASSES = 36

def log(message):
    if args.verbose:
        print(message)
    return

def idx2class(idx):
    return IDX2CLASS[idx]

def load_data(test_fpath, config):
    """ Loads transformed data """

    config["n_channels"] = 3

    test_transforms = []

    # resizing images
    test_transforms.append(transforms.Resize((config["resize"], config["resize"])))

    # convert to grayscale - compulsory
    test_transforms.append(transforms.Grayscale(config["n_channels"]))

    # conversion into pytorch tensor - compulsory
    test_transforms.append(transforms.ToTensor())

    # normalize - good for compatibility with inceptionv3
    if config["normalize"]:
        if config["n_channels"] == 3:
            test_transforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        else:
            test_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    # create image transformations list
    test_transforms = transforms.Compose(test_transforms)

    log("Loading Dataset")
    tfold = None #"test" if ("fold" in pd.read_csv(test_fpath).columns) else None
    test_data = ImageDataset(data_file=test_fpath, fpath_col_name=FPATH_COL_NAME, lbl_col_name=LBL_COL_NAME, fold=tfold, class_vals=CLASSES, transforms=test_transforms, class_mapping=CLASS_MAPPING)
    log("{} test images found belonging to {} classes".format(len(test_data), len(test_data.classes)))

    log("Preparing batch loaders from datasets")
    test_data_loader = DataLoader(test_data, 128)
    return test_data_loader

def load_model(config, weights="best_weights.pth", device='cpu'):
    """ Loads a pretrained InceptionV3 model, which is then updated using fine-tuning methods """

    # load model configuration
    if "model" in config.keys():
        name = config["model"]
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
    elif name == "vgg11":
        model = torchvision.models.vgg11_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, N_CLASSES)
    elif name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, N_CLASSES)
    elif name == "simple":
        from custom_models import multiClassPerceptron
        model = multiClassPerceptron(in_channels=3*299*299, hidden_layers=[], out_channels=N_CLASSES)
    elif name == "densenet169":
        model = torchvision.models.densenet169(pretrained=True)
        model.classifier = nn.Linear(1664, N_CLASSES)
    elif name == "efficient-net-b3":
        model = torchvision.models.efficientnet_b3(pretrained=True)
        model.classifier = nn.Linear(1536, N_CLASSES)
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
        true_labels = np.array(list(map(idx2class, labels.cpu().detach().numpy())))
        pred_labels = np.array(list(map(idx2class, final_preds.cpu().detach().numpy())))
        row_result = np.concatenate([np.array(fpath)[:, None], true_labels[:, None], pred_labels[:, None], raw_preds.cpu().detach().numpy()], axis=1)
        row_df = pd.DataFrame(row_result, columns=columns)
        results_df = pd.concat([results_df, row_df])

    # statistics averaged over entire training set
    acc = running_corrects.double() / test_data_size
    log("Model Accuracy: {:.4f}".format(acc))
    log("Saving Results")
    results_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    return results_df

def create_confusion(results_df, save_dir):
    log("Creating confusion matrix...")
    # create matrix
    predicted_labels = results_df["Predicted Class"].values
    true_labels = results_df["True Class"].values
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(true_labels, predicted_labels)
    with open(os.path.join(save_dir, "cm.npy"), 'wb') as f:
        np.save(f, confusion_matrix)

    # visualize matrix
    plt.figure(figsize=(25, 12))
    ax = sns.heatmap(confusion_matrix, annot=True, fmt="g", xticklabels=CLASS_LIST, yticklabels=CLASS_LIST, cmap="YlGnBu")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    return

def calc_accuracies(df, save_dir):

    df1 = pd.read_csv(os.path.join(save_dir, "predictions.csv"))

    classes = ["ABCA4", "BBS1", "BEST1", "CACNA1F", "CDH23", "CERKL", "CHM", "CNGA3", "CNGB3",
           "CRB1", "CRX", "CYP4V2", "EFEMP1", "EYS", "GUCY2D", "KCNV2", "MERTK", "MTTL1",
           "MYO7A", "NR2E3", "OPA1", "PDE6B", "PROML1", "PRPF31", "PRPF8", "PRPH2", "RDH12",
           "RHO", "RP1", "RP1L1", "RP2", "RPE65", "RPGR", "RS1", "TIMP3", "USH2A"]

    idx2class = {i: cls for i, cls in enumerate(classes)}
    idx_mapper = lambda i: idx2class[i]

    true_classes = df1["True Class"]
    pandas_query = ['Prob_' + cls for cls in classes]
    per_class_probs = df1[pandas_query].values
    per_class_probs = np.argsort(per_class_probs, axis=1)

    accuracies = {}
    for top in [1, 2, 3, 5, 10, 20, 36]:
        acc = 0
        for i, c in enumerate(true_classes):
            if c in map(idx_mapper, per_class_probs[i, -top:]):
                acc += 1
            else:
                acc += 0
        acc = acc / len(df1)
        accuracies["Top-{}".format(top)] = acc
    
    with open(os.path.join(save_dir, "accuracies.json"), 'w') as f:
        json.dump(accuracies, f)

    return accuracies

if __name__ == "__main__":

    # for parsing command-line arguments
    parser = argparse.ArgumentParser()
    # dataset-related
    parser.add_argument('--config', help="Neural Network model configuration in json file")
    parser.add_argument('--weights', help="Path to weights file for trained model (.pth file)", type=str)
    parser.add_argument('--test', help="Provide a csv file path to the validation set images", type=str)
    parser.add_argument('--confusion', help="Create confusion matrix", default=1)
    parser.add_argument('--top-acc', help="Compute top-X accuracies", default=1)
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

    # load experiment config
    config = json.load(open(args.config))

    # create dataloaders
    dataloader = load_data(args.test, config)

    # load model
    model = load_model(config, args.weights, device)

    # begin model inference
    results = predict(model, dataloader, device=device, save_dir=args.save_dir)

    # results = pd.read_csv(os.path.join(args.save_dir, "predictions.csv"))

    # optionally create results
    if args.confusion:
        create_confusion(results, args.save_dir)

    if args.top_acc:
        calc_accuracies(results, args.save_dir)
