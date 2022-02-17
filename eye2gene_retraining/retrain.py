# import libraries
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
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

def dump_history(history, save_dir, fname):
    """ Saves history in json file in save dir under provided fname """
    with open(os.path.join(save_dir, fname), 'w') as f:
        json.dump(history, f)

def plot_performance(history, save_dir):
    """ saves learning curves of loss and accuracy metrics """
    for metric in ["Loss", "Acc"]:
        plt.figure(figsize=(12, 6))
        plt.plot(history["Train"][metric], color="blue")
        plt.plot(history["Val"][metric], color="red")
        plt.legend(["Train", "Val"])
        plt.title("{} over epochs".format(metric))
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.savefig(os.path.join(save_dir, "{}.jpg".format(metric)))
        plt.close()

# set global variables
FPATH_COL_NAME = "file.path"
LBL_COL_NAME = "gene"
CLASSES = "../classes.txt"
CLASS_MAPPING = "../classes_mapping.json"
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = lambda x, y, z: optim.Adam(x, lr=y, weight_decay=z)
N_CLASSES = 36
IM_SIZE = 299

def save_config(args):
    with open(os.path.join(args.save_dir, "model_config.json"), 'w') as f:
        json.dump(vars(args), f)
    return

def log(message):
    if args.verbose:
        print(message)
    return

def load_data(train_fpath, val_fpath, **kwargs):
    """ Loads transformed data """

    train_transforms = []
    val_transforms = []

    # resizing images
    if "resize" in kwargs:
        train_transforms.append(transforms.Resize((IM_SIZE, IM_SIZE)))
        val_transforms.append(transforms.Resize((IM_SIZE, IM_SIZE)))

    # convert to grayscale - compulsory
    train_transforms.append(transforms.Grayscale(3))
    val_transforms.append(transforms.Grayscale(3))

    # random hflip - optional, but better for augmentation purposes 
    if "random_hflip" in kwargs:
        train_transforms.append(transforms.RandomHorizontalFlip(p=0.3))

    if "brightness" in kwargs:
        train_transforms.append(transforms.ColorJitter(brightness=(0.5, 1.5)))

    if "rotation" in kwargs:
        train_transforms.append(transforms.RandomAffine(degrees=15))

    # conversion into pytorch tensor - compulsory
    train_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.ToTensor())

    # normalize - good for compatibility with inceptionv3
    if "normalize" in kwargs:
        train_transforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        val_transforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

    # create image transformations list
    train_transforms = transforms.Compose(train_transforms)
    val_transforms = transforms.Compose(val_transforms)

    log("Loading Datasets")
    tfold = "train" if ("fold" in pd.read_csv(train_fpath).columns and len(pd.read_csv(train_fpath).fold.unique()) <= 4) else None
    vfold = "val" if ("fold" in pd.read_csv(val_fpath).columns) else None
    train_data = ImageDataset(data_file=train_fpath, fpath_col_name=FPATH_COL_NAME, lbl_col_name=LBL_COL_NAME, fold=tfold, class_vals=CLASSES, transforms=train_transforms, class_mapping=CLASS_MAPPING)
    val_data = ImageDataset(data_file=val_fpath, fpath_col_name=FPATH_COL_NAME, lbl_col_name=LBL_COL_NAME, fold=vfold, class_vals=CLASSES, transforms=val_transforms, class_mapping=CLASS_MAPPING)
    log("{} training images found belonging to {} classes".format(len(train_data), len(train_data.classes)))
    log("{} validation images found belonging to {} classes".format(len(val_data), len(val_data.classes)))

    log("Preparing batch dataloaders from datasets")
    train_data_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, args.batch_size)
    dataloaders = {"Train": train_data_loader, "Val": val_data_loader}
    return dataloaders

def load_model(name="inceptionv3", device='cpu'):
    """ Loads a pretrained InceptionV3 model, which is then updated using fine-tuning methods """
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
    elif name == "simple":
        from custom_models import multiClassPerceptron
        model = multiClassPerceptron(in_channels=3*299*299, hidden_layers=[], out_channels=N_CLASSES)
    else:
        raise ValueError("Model can only be `inceptionv3`, `vgg16` or `resnet18`")

    # push model to desired device (only if gpu's are provided)
    if isinstance(device, list):
        model.to(device[0])
        model = nn.DataParallel(model, device_ids=device)
    else:
        model.to(torch.device("cuda:{}".format(device)))
    return model

def train_model(dataloaders, model, criterion, optimizer, num_epochs, device='cpu', save_dir=None, save_best_weights=False, w_monitor="loss", early_stopping=True, es_monitor="loss", epsilon=0.05, tolerance=10, is_inception=True):

    if isinstance(device, list):
        base_device = torch.device("cuda:{}".format(device[0]))
    elif isinstance(device, int):
        base_device = torch.device("cuda:{}".format(device))
    else:
        base_device = device
        
    # start time
    since = time.time()

    # store dataset sizes
    train_data_size = len(dataloaders["Train"].dataset)
    val_data_size = len(dataloaders["Val"].dataset)
    dataset_sizes = {"Train": train_data_size, "Val": val_data_size}
    
    # initializing some variables which will be important for weight saving and early stopping
    k = 0
    best_loss = float("inf")
    epoch_prev_loss, epoch_cur_loss = float("inf"), float("inf") 
    best_acc = 0.0
    epoch_prev_acc, epoch_cur_acc = 0, 0

    # history dictionary
    history = {"Train": {"Loss": [], "Acc": []}, "Val": {"Loss":[], "Acc": []}}

    # begin training
    for epoch in range(num_epochs):
        log("Epoch {}/{}".format(epoch+1, args.epochs))

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for _, _, inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(base_device)
                labels = labels.to(base_device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    if is_inception and phase == "Train":
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics per batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # statistics averaged over entire training set
            epoch_cur_loss = running_loss / dataset_sizes[phase]
            epoch_cur_acc = running_corrects.double() / dataset_sizes[phase]
            # update history dictionary
            history[phase]["Loss"].append(epoch_cur_loss)
            history[phase]["Acc"].append(epoch_cur_acc.item())
            log("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_cur_loss, epoch_cur_acc))
            # dump latest results - history json and learning plots
            dump_history(history, save_dir, "training_logs.json")
            plot_performance(history, save_dir)

            if phase == "Val":
                # =====================================================================================
                # saving best weights can be based on the metric we are monitoring (val loss/accuracy)
                # =====================================================================================
                if save_best_weights:
                    if w_monitor == "loss":
                        if epoch_cur_loss < best_loss:
                            log("Loss decreased from {} to {}, updating model weights\n".format(best_loss, epoch_cur_loss))
                            best_loss = epoch_cur_loss
                            if isinstance(device, list):
                                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_weights.pth"))
                            else:
                                torch.save(model.state_dict(), os.path.join(save_dir, "best_weights.pth"))
                        else:
                            log("Current loss is not smaller than the best loss, no weight update.\n")
                    elif w_monitor == "acc":
                        if epoch_cur_acc > best_acc:
                            log("Accuracy increased from {} to {}, updating model weights\n".format(best_acc, epoch_cur_acc))
                            best_acc = epoch_cur_acc
                            if isinstance(device, list):
                                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_weights.pth"))
                            else:
                                torch.save(model.state_dict(), os.path.join(save_dir, "best_weights.pth"))
                        else:
                            log("Current accuracy is not larger than the best accuracy, no weight update.\n")
                    else:
                        raise ValueError("w_monitor can only be `loss` or `acc`.")
                else:
                    log("Saving latest weights to {}\n".format(os.path.join(save_dir, "best_weights.pth")))
                    if isinstance(device, list):
                        torch.save(model.module.state_dict(), os.path.join(save_dir, "best_weights.pth"))
                    else:
                        torch.save(model.state_dict(), os.path.join(save_dir, "best_weights.pth"))

                # =================================================================================================================
                # early stopping generally uses val accuracy but I've included functionality to allow for either val loss/accuracy
                # =================================================================================================================
                if early_stopping:
                    if es_monitor == "loss":
                        if np.abs(epoch_cur_loss - epoch_prev_loss) <= epsilon:
                            if (k < tolerance):
                                k += 1
                            else:
                                log("Minimal change in loss. Training completed early!")
                                # Print time elapsed and summary of training
                                time_elapsed = time.time() - since
                                log("Summary\n Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
                                log("Best Validation Loss: {:4f}".format(best_loss))
                                log("Saved Weights to: {}".format(os.path.join(save_dir, "best_weights.pth")))
                                return
                        else:
                            k = 0
                    elif es_monitor == "acc":
                        if np.abs(epoch_cur_acc - epoch_prev_acc) <= epsilon:
                            if (k < tolerance):
                                k += 1
                            else:
                                log("Minimal change in accuracy. Training completed early!")
                                # Print time elapsed and summary of training
                                time_elapsed = time.time() - since
                                log("Summary\n Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
                                log("Best Validation Accuracy: {:4f}".format(best_acc))
                                log("Saved Weights to: {}".format(os.path.join(save_dir, "best_weights.pth")))
                                return
                        else:
                            k = 0
                    else:
                        raise ValueError("es_monitor can only be `loss` or `acc`.")

        epoch_prev_loss = epoch_cur_loss
        epoch_prev_acc = epoch_cur_acc

    # Print time elapsed and summary of training
    time_elapsed = time.time() - since
    log("Summary\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    log("Saved Weights to: {}".format(os.path.join(save_dir, "best_weights.pth")))
    return 

if __name__ == "__main__":

    # for parsing command-line arguments
    parser = argparse.ArgumentParser()
    # dataset-related
    parser.add_argument('--model', default="inceptionv3", help="Neural Network model", choices=["inceptionv3", "resnet18", "vgg16"])
    parser.add_argument('--train', help="Provide a csv file path to the training images", type=str)
    parser.add_argument('--val', help="Provide a csv file path to the validation set images", type=str)
    parser.add_argument('--resize', default=299, help="Desired dimension to resize image to", type=int)
    parser.add_argument('--random-hflip', help="Adds a horizontal flip with probability p=0.3", default=1)
    parser.add_argument('--normalize', help="Normalizes images with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]", default=1)
    parser.add_argument('--brightness', help="Adds brightness to images", default=1)
    parser.add_argument('--rotation', help="Adds rotation to the images", default=1)
    # training hyperparams
    parser.add_argument('--epochs', default=100, help="Number of epochs", type=int)
    parser.add_argument('--batch-size', default=32, help="Batch size parameter", type=int)
    parser.add_argument('--lr', default=1e-3, help="Learning rate parameter", type=float)
    parser.add_argument('--weight-decay', default=1, help="Weight decay parameter for L2 regularization", type=float)
    # other training params 
    parser.add_argument('--save-best', default=0, help="Whether to save the best weights or the last weights")
    parser.add_argument('--w-monitor', default="loss", help="Metric to monitor during training and update parameters based on", choices=["loss", "acc"])
    parser.add_argument('--es', default=1, help="Whether to use Early Stopping")
    parser.add_argument('--es-monitor', default="loss", help="Metric to monitor for Early Stopping", choices=["loss", "acc"])
    parser.add_argument('--epsilon', default=0.005, help="Expected difference between consecutive epochs to halt training early.")
    parser.add_argument('--tolerance', default=10, help="Expected number of consecutive epochs for which performance difference is smaller than epsilon")
    parser.add_argument('--save-dir', default="experiment_results/", help="Name of directory in which to save experiment results and logs.", type=str)
    parser.add_argument('--save-logs', help="Whether to save the logs in the save dir or not.", action="store_true")
    parser.add_argument('-v', '--verbose', help="Prints logs of the progress during model training", action="store_true")
    parser.add_argument('-d', '--device', default="cpu", help="Devices on which model is trained", type=str)
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

    # make training results directory if doesn't exist
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    # save training configuration
    save_config(args)

    # create dataloaders
    dataloaders = load_data(args.train, args.val, **vars(args))

    # load model
    model = load_model(args.model, device)

    # begin model training
    train_model(dataloaders, model, CRITERION, optimizer=OPTIMIZER(model.parameters(), args.lr, args.weight_decay), num_epochs=args.epochs, device=device, save_dir=args.save_dir, save_best_weights=args.save_best, w_monitor=args.w_monitor, early_stopping=args.es, es_monitor=args.es_monitor, epsilon=args.epsilon, tolerance=args.tolerance)
