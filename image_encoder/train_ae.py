# import essential libraries
import os 
import sys
sys.path.append("..")

import torch 
import json
from torch import nn
from torchvision import transforms
from utils.data_utils import ImageDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import set_seed, logger
import matplotlib.pyplot as plt 
import argparse
from ae_model import ConvolutionalAE
from torchvision.utils import make_grid

def get_devices(device):
    if device == "cpu" or not torch.cuda.is_available():
        print("Training on cpu")
        device = args.device
    elif len(device) == 1:
        device = int(args.device)
        print("Found 1 GPU : cuda:{}".format(device))
    else:
        device = list(map(int, args.device.split(',')))
        print("Found {} GPUs : ".format(len(device)) + ", ".join(["cuda:{}".format(x) for x in device]))
    return device

def save_config(args):
    with open(os.path.join(args.save_dir, "model_config.json"), 'w') as f:
        json.dump(vars(args), f)
    return

def plot_performance(history, save_dir, save_as):
    """ saves learning curve """
    plt.figure(figsize=(12, 6))
    for phase in ["Train", "Val"]:
        plt.plot(history[phase])
    plt.legend(["Train", "Val"])
    plt.title("Reconstruction loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, save_as))
    plt.close()

def visualize_reconstruction(image1, image2, save_dir, save_as):
    ''' Saves image grid of 5 sample reconstructions '''
    # print(image1.shape)
    image_pairs = list(zip((image1[:5] + 1)/2, (image2[:5] + 1)/2))
    # print(len(image_pairs), image_pairs[0][0].shape)
    image_pairs = [torch.cat((pair[0][None, :, :, :], pair[1][None, :, :, :]), 0) for pair in image_pairs]
    # print(image_pairs[0].shape)
    image_pairs = torch.cat(image_pairs, 0)
    # print(image_pairs.shape)
    assert image_pairs.shape[0] == 10
    image_grid = make_grid(image_pairs, nrow=2)

    plt.figure(figsize=(6, 12))
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.title("Sample Image Reconstructions")
    plt.savefig(os.path.join(save_dir, save_as))
    plt.close()

def dump_history(history, save_dir):
    """ Saves history in json file in save dir under provided fname """
    with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f)

def load_datasets(training_fpath, val_fpath, im_resize, b_size):
    
    # set global variables
    with open("../classes.txt") as f:
        classes = f.read().splitlines()
    
    # prepare data transformations
    train_transforms = transforms.Compose([transforms.Resize((im_resize, im_resize)), transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    val_transforms = transforms.Compose([transforms.Resize((im_resize, im_resize)), transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])
    
    # prepare datasets
    train_data = ImageDataset(training_fpath, "file.path", "gene", classes, train_transforms)
    val_data = ImageDataset(val_fpath, "file.path", "gene", classes, val_transforms)
    
    # prepare dataloaders
    train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=b_size, shuffle=True)
    
    return train_dataloader, val_dataloader

def train_model(dataloaders, model, criterion, optimizer, num_epochs, save_step, save_dir, device="cpu", verbose=True):
    
    # set logger function
    log = lambda x: logger(x, verbose=verbose)
    
    # push model to device
    if isinstance(device, list):
        base_device = torch.device("cuda:" + str(device[0]))
        model.to(base_device)
        model = nn.DataParallel(model, device_ids=device)
    elif isinstance(device, int):
        base_device = torch.device("cuda:"+str(device))
        model.to(base_device)
    else:
        base_device = device

    # store dataset sizes
    train_data_size = len(dataloaders["Train"].dataset)
    val_data_size = len(dataloaders["Val"].dataset)
    dataset_sizes = {"Train": train_data_size, "Val": val_data_size}

    # history dictionary
    history = {"Train": [], "Val": []}

    # metrics
    best_loss = float("inf")

    # begin training
    for epoch in range(num_epochs):
        log("Epoch {}/{}".format(epoch+1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            # Iterate over data.
            for _, _, inputs, _ in tqdm(dataloaders[phase]):
                inputs = inputs.to(base_device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics per batch
                running_loss += loss.item() * inputs.size(0)

            # statistics averaged over entire training set
            epoch_cur_loss = running_loss / dataset_sizes[phase]
            
            # update history dictionary
            history[phase].append(epoch_cur_loss)
            log("{} Loss: {:.4f}".format(phase, epoch_cur_loss))

            # save weights
            if phase == "Val":
                if epoch_cur_loss < best_loss:
                    log("Loss has reduced from {} to {}. Updating weights".format(best_loss, epoch_cur_loss))
                    best_loss = epoch_cur_loss
                    if isinstance(device, list):
                        torch.save(model.module.state_dict(), os.path.join(save_dir, "best_weights.pth"))
                    else:
                        torch.save(model.state_dict(), os.path.join(save_dir, "best_weights.pth"))

        # visualize metrics and image reconstructions
        if (epoch + 1) % save_step == 0 or epoch == 0:
            # plot performance
            plot_performance(history, save_dir, "losses.png")
            # visualize image reconstruction
            visualize_reconstruction(inputs.detach().cpu(), outputs.detach().cpu(), save_dir, "reconstruction_epoch_{}.png".format(epoch+1))
    
    return history

if __name__ == "__main__":
    # for parsing command-line arguments
    parser = argparse.ArgumentParser()

    # dataset-related
    parser.add_argument('--train-fpath', help="Provide a csv file path to the training images", type=str)
    parser.add_argument('--val-fpath', help="Provide a csv file path to the validation set images", type=str)

    # training hyperparams
    parser.add_argument('--resize', default=512, help="Dimensions of resized image", type=int)
    parser.add_argument('--latent-size', help="Size of feature vector to learn with autoencoder", default=256, type=int)
    parser.add_argument('--epochs', default=100, help="Number of epochs", type=int)
    parser.add_argument('--loss', default="mse", help="Loss function for Autoencoder")
    parser.add_argument('--batch-size', default=128, help="Batch size parameter", type=int)
    parser.add_argument('--lr', default=1e-3, help="Learning rate parameter", type=float)

    # other training params
    parser.add_argument('--save-step', default=1, help="Save performance and image reconstructions", type=int)
    parser.add_argument('--save-dir', default="results/", help="Name of directory in which to save experiment results and logs.", type=str)
    parser.add_argument('-v', '--verbose', help="Prints logs of the progress during model training", action="store_false")
    parser.add_argument('-d', '--device', help="Devices on which model is trained", type=str)
    parser.add_argument('--seed', help="To ensure reproducibility of results", type=int, default=1399)
    args = parser.parse_args()

    # ensures reproducibility
    set_seed(args.seed)
    log = lambda x: logger(x, verbose=args.verbose)

    # make training results directory if doesn't exist
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    # save config
    save_config(args)

    # get devices
    args.device = get_devices(args.device)
    print(args.device)

    # create dataloaders
    dataloaders = load_datasets(args.train_fpath, args.val_fpath, im_resize=args.resize, b_size=args.batch_size)
    log("Found {} training images belonging to {} classes".format(len(dataloaders[0].dataset), dataloaders[0].dataset.n_classes))
    log("Found {} test images belonging to {} classes\n".format(len(dataloaders[1].dataset), dataloaders[1].dataset.n_classes))

    # load model
    log("Loading Model:\n")
    model = ConvolutionalAE(im_size=args.resize, latent_size=args.latent_size)
    log(model)
    
    # load other components
    criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # begin model training
    log("\nBegin Model Training:\n")
    training_history = train_model({"Train":dataloaders[0], "Val":dataloaders[1]}, model, criterion, optimizer, args.epochs, args.save_step, args.save_dir, args.device, args.verbose)

    # save training history
    dump_history(training_history, args.save_dir)