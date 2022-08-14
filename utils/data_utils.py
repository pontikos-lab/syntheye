"""
This module contains all the helper functions for dealing with image data and model
"""

# import libraries
import os
import sys
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from PIL import Image

def get_one_hot_labels(labels, n_classes):
    """
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    """
    return F.one_hot(labels, n_classes)


def combine_vectors(x, y):
    """
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector.
        In this assignment, this will be the noise vector of shape (n_samples, z_dim),
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    """
    combined = torch.cat((x, y), dim=1)
    return combined

class ImageDataset(Dataset):

    """ PyTorch class for Dataset """

    def __init__(self, data_file: str, fpath_col_name: str, lbl_col_name=None, transforms=None, class_mapping=None, class_vals=None, fold=None):
        
        # read dataframe
        df = pd.read_csv(data_file)

        # if labels are provided
        if lbl_col_name is not None:

            if class_vals is None:

                if fold is None:
                    self.img_dir = list(df[fpath_col_name])
                    self.img_labels = list(df[lbl_col_name])

                elif fold == "train":
                    train_df = df.where(df.fold.isin([1, 2, 3, 4])).dropna()
                    self.img_dir = list(train_df[fpath_col_name])
                    self.img_labels = list(train_df[lbl_col_name])

                elif fold == "val":
                    val_df = df.where(df.fold == 0).dropna()
                    self.img_dir = list(val_df[fpath_col_name])
                    self.img_labels = list(val_df[lbl_col_name])

                elif fold == "test":
                    test_df = df.where(df.fold == -1).dropna()
                    self.img_dir = list(test_df[fpath_col_name])
                    self.img_labels = list(test_df[lbl_col_name])

                else:
                    raise Exception("fold can be train, val or test only.")

            else:

                # load selected classes
                if isinstance(class_vals, str):
                    with open(class_vals, 'r') as f:
                        selected_classes = f.read().splitlines()
                elif isinstance(class_vals, list):
                    selected_classes = class_vals
                else:
                    raise ValueError("Class values parameters can be a filepath or a list of class values!")

                # get rows of dataframe for selected classes
                df_subset = df.loc[df[lbl_col_name].isin(selected_classes)]

                if fold is None:
                    self.img_dir = list(df_subset[fpath_col_name])
                    self.img_labels = list(df_subset[lbl_col_name])
                elif fold == "train":
                    train_df = df_subset.where(df_subset.fold.isin([1, 2, 3, 4])).dropna()
                    self.img_dir = list(train_df[fpath_col_name])
                    self.img_labels = list(train_df[lbl_col_name])
                elif fold == "val":
                    val_df = df_subset.where(df_subset.fold == 0).dropna()
                    self.img_dir = list(val_df[fpath_col_name])
                    self.img_labels = list(val_df[lbl_col_name])
                elif fold == "test":
                    test_df = df_subset.where(df_subset.fold == -1).dropna()
                    self.img_dir = list(test_df[fpath_col_name])
                    self.img_labels = list(test_df[lbl_col_name])
                else:
                    raise Exception("fold can be train, val, or test only.")

        else:
            if fold is None:
                self.img_dir = list(df[fpath_col_name])

            elif fold == "train":
                train_df = df.where(df.fold != -1).dropna()
                self.img_dir = list(train_df[fpath_col_name])

            elif fold == "test":
                test_df = df.where(df.fold == -1).dropna()
                self.img_dir = list(test_df[fpath_col_name])

            else:
                raise Exception("fold can be train, val, or test only.")

            self.img_labels = None

        # determine classes and mappings from dataset or from a provided dictionary json file
        if self.img_labels is not None:

            if class_mapping is not None:

                if isinstance(class_mapping, str):
                    assert class_mapping.endswith(".json"), "Must provide a json file!"
                    self.class2idx = json.load(open(class_mapping))
                else:
                    self.class2idx = class_mapping
                
                self.idx2class = {v:k for (k,v) in self.class2idx.items()}
                self.classes = list(self.class2idx.keys()) if class_vals is None else class_vals
                self.n_classes = len(self.classes)
            
            else:
                # relevant attributes if classes are provided
                self.class2idx = dict(zip(self.classes, range(self.n_classes)))
                self.idx2class = dict(zip(range(self.n_classes), self.classes))
                self.classes = list(np.unique(self.img_labels))
                self.n_classes = len(self.classes)

        else:
            self.classes = None
            self.n_classes = None
            self.idx2class = None
            self.class2idx = None

        # image transformations list
        self.transform = transforms

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, item):
        # create PIL object of item-th image
        image = Image.open(self.img_dir[item])

        # get the label index for the item-th image - this is just -1 if no labels are found in datafile
        label = self.class2idx[self.img_labels[item]] if self.img_labels is not None else -1

        # transform images
        if self.transform is not None:
            image = self.transform(image)

        return item, self.img_dir[item], image, label
