import os, sys, json
sys.path.append('..')

# import libraries
import torch
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data_utils import ImageDataset
from trainScript import Trainer
from clutils import logger, save_config, set_seed

# set global variables
MODEL_CHOICES = ["inceptionv3", "resnet18", "vgg16", "simple", "alexnet", "vgg11", "efficient-net-b3", "densenet169", "simple-convnet", "feature-based", "vit"]
FPATH_COL_NAME = "file.path"
LBL_COL_NAME = "gene"
CLASS_MAPPING = "../classes_mapping.json"
CRITERION = torch.nn.CrossEntropyLoss()
OPTIMIZER = lambda x, y, z: torch.optim.Adam(x, lr=y, weight_decay=z)

def parse_args() -> dict:
    """ Parses command line arguments """
    import argparse
        # for parsing command-line arguments
    parser = argparse.ArgumentParser()
    # dataset-related args
    parser.add_argument('--model', default="inceptionv3", help="Neural Network model", choices=MODEL_CHOICES)
    parser.add_argument('--train', help="Provide a csv file path to the training images", type=str)
    parser.add_argument('--val', help="Provide a csv file path to the validation set images", type=str)
    parser.add_argument('--resize', default=299, help="Desired dimension to resize image to", type=int)
    parser.add_argument('--n-channels', default=3, help="Preprocess images to be 3 or 1 channel", type=int)
    parser.add_argument('--random-hflip', help="Adds a horizontal flip with probability p=0.3", default=1)
    parser.add_argument('--normalize', help="Normalizes images with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]", default=1)
    parser.add_argument('--brightness', help="Adds brightness to images", default=1)
    parser.add_argument('--rotation', help="Adds rotation to the images", default=1)
    # training hyperparams
    parser.add_argument('--epochs', default=100, help="Number of epochs", type=int)
    parser.add_argument('--batch-size', default=32, help="Batch size parameter", type=int)
    parser.add_argument('--lr', default=1e-3, help="Learning rate parameter", type=float)
    parser.add_argument('--weight-decay', default=1, help="Weight decay parameter for L2 regularization", type=float)
    # other params 
    parser.add_argument('--save-best', default=0, help="Whether to save the best weights or the last weights")
    parser.add_argument('--monitor', default="loss", help="Metric to monitor during training and update parameters based on", choices=["loss", "acc"])
    # parser.add_argument('--es', default=1, help="Whether to use Early Stopping")
    # parser.add_argument('--es-monitor', default="loss", help="Metric to monitor for Early Stopping", choices=["loss", "acc"])
    # parser.add_argument('--epsilon', default=0.005, help="Expected difference between consecutive epochs to halt training early.")
    # parser.add_argument('--tolerance', default=10, help="Expected number of consecutive epochs for which performance difference is smaller than epsilon")
    parser.add_argument('--save-dir', default="experiment_results/", help="Name of directory in which to save experiment results and logs.", type=str)
    parser.add_argument('--save-logs', help="Whether to save the logs in the save dir or not.", action="store_true")
    parser.add_argument('-v', '--verbose', help="Prints logs of the progress during model training", action="store_true")
    parser.add_argument('-d', '--device', default="cpu", help="Devices on which model is trained", type=str)
    parser.add_argument('--seed', help="To ensure reproducibility of results", type=int)
    args = parser.parse_args()
    return args

def load_data(train_fpath: str, val_fpath: str, **kwargs) -> dict:
    
    """ Loads transformed data """

    CLASSES = list(json.load(open(CLASS_MAPPING)).keys())

    train_transforms = []
    val_transforms = []

    # resizing images - compulsory 
    if "resize" in kwargs:
        train_transforms.append(transforms.Resize((kwargs["resize"], kwargs["resize"])))
        val_transforms.append(transforms.Resize((kwargs["resize"], kwargs["resize"])))

    # convert to grayscale - compulsory
    train_transforms.append(transforms.Grayscale(kwargs["n_channels"]))
    val_transforms.append(transforms.Grayscale(kwargs["n_channels"]))

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
    if kwargs["normalize"]:
        if kwargs["n_channels"] == 3:
            train_transforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            val_transforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        else:
            train_transforms.append(transforms.Normalize(0.5, 0.5))
            val_transforms.append(transforms.Normalize(0.5, 0.5))

    # create image transformations list
    train_transforms = transforms.Compose(train_transforms)
    val_transforms = transforms.Compose(val_transforms)

    log("Loading Datasets")

    tfold = None #"train" if ("fold" in pd.read_csv(train_fpath).columns and len(pd.read_csv(train_fpath).fold.unique()) <= 4) else None
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

def load_model(model_name: str) -> torch.nn.Module:
    
    """ Loads deep learning model """

    log("Load pretrained {}".format(model_name))

    n_classes = len(json.load(open(CLASS_MAPPING)).keys())

    if model_name == "inceptionv3":
        model = torchvision.models.inception_v3(pretrained=True)
        # edit the last layer to have 36 layers only
        model.AuxLogits.fc = torch.nn.Linear(768, n_classes)
        model.fc = torch.nn.Linear(2048, n_classes)

    elif model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, n_classes)

    elif model_name == "vgg16":
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, n_classes)

    elif model_name == "vgg11":
        model = torchvision.models.vgg11_bn(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, n_classes)

    elif model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, n_classes)

    elif model_name == "simple":
        from custom_models import multiClassPerceptron
        model = multiClassPerceptron(in_channels=1*299*299, hidden_layers=[], out_channels=n_classes)

    elif model_name == "densenet169":
        model = torchvision.models.densenet169(pretrained=True)
        model.classifier = torch.nn.Linear(1664, n_classes)

    elif model_name == "efficient-net-b3":
        model = torchvision.models.efficientnet_b3(pretrained=True)
        model.classifier = torch.nn.Linear(1536, n_classes)

    elif model_name == "simple-convnet":
        from custom_models import simpleConvNet
        model = simpleConvNet()

    elif model_name == "feature-based":
        from custom_models import featureBased
        model = featureBased()

    elif model_name == "vit":
        from vit_pytorch import ViT
        model = ViT(image_size=512, patch_size=16, num_classes=n_classes, dim=1024, mlp_dim=2048, depth=6, heads=16, channels=1)

    else:
        raise ValueError("Model can only be `inceptionv3`, `vgg16` or `resnet18`")

    return model

if __name__ == "__main__":

    # parse command line args
    args = parse_args()
    log = lambda m: logger(m, args.verbose)

    # load the device name
    if args.device == "cpu" or not torch.cuda.is_available():
        log("Training on cpu")
        device = args.device
    elif len(args.device) == 1:
        device = int(args.device)
        log("Found 1 GPU : cuda:{}".format(device))
    else:
        device = list(map(int, args.device.split(',')))
        log("Found {} GPUs : ".format(len(device)) + ", ".join(["cuda:{}".format(x) for x in device]))

    # set seed for reproducibility
    set_seed(args.seed)

    # make training results directory if doesn't exist
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    # save training configuration
    save_config(args, "model_config.json")

    # create dataloaders
    dataloaders = load_data(args.train, args.val, **vars(args))

    # load model and optimizer
    model = load_model(args.model)
    optimizer = OPTIMIZER(model.parameters(), args.lr, args.weight_decay)

    # load model trainer
    trainer = Trainer(model, device)

    # begin training
    if args.model == "inceptionv3":
        trainer.train(dataloaders, CRITERION, optimizer, args.epochs, args.save_dir, args.save_best, args.monitor, is_inception=True)
    else:
        trainer.train(dataloaders, CRITERION, optimizer, args.epochs, args.save_dir, args.save_best, args.monitor)