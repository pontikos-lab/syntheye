""" Classifier Training Utilities """

# import libraries
import os
import json

def logger(message: str, verbose: bool = True) -> None:
    """ Prints message if verbosity enabled """
    if verbose:
        print(message)

def save_config(args: dict, save_as: str) -> None:
    """ Save configuration given dictionary of parameter-value pairs """
    with open(os.path.join(args.save_dir, save_as), 'w') as f:
        json.dump(vars(args), f)

def set_seed(seed: int) -> None:
    """ Set random seed for reproducibility of experiments """
    import torch 
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)

class History():
    """ Logs the training history """

    def __init__(self):
        self.data = {"Train": {"Loss": [], "Acc": []}, "Val": {"Loss":[], "Acc": []}}

    def add_entry(self, phase, loss, acc):
        self.data[phase]["Loss"].append(loss) 
        self.data[phase]["Acc"].append(acc)

    def plot(self, save_to: str) -> None:
        """ Saves learning curves of loss and accuracy metrics """
        import matplotlib.pyplot as plt
        for metric in ["Loss", "Acc"]:
            plt.figure(figsize=(12, 6))
            plt.plot(self.data["Train"][metric], color="blue")
            plt.plot(self.data["Val"][metric], color="red")
            plt.legend(["Train", "Val"])
            plt.title("{} over epochs".format(metric))
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.savefig(os.path.join(save_to, "{}.jpg".format(metric)))
            plt.close()

    def dump(self, save_to: str) -> None:
        with open(save_to, 'w') as f:
            json.dump(self.data, f)
