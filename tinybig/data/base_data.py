# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Base Dataset and Dataloader #
###############################

"""
The base dataloader and dataset classes.

This module implements the base dataloader class and dataset class,
which can be used for loading datasets for RPN model training and testing in tinyBIG toolkit.
"""
from abc import abstractmethod
import numpy as np
from typing import Union, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from tinybig.config.base_config import config

class dataloader:
    """
    The base dataloader class.

    It defines the base dataloader class that can be used for loading data from files.

    Attributes
    ----------
    name: str, default = 'base_dataloader'
        The name of the base dataloader class.

    Methods
    ----------
    __init__
        The base dataloader class initialization method.

    load
        The load method for loading the data from file.
    """
    def __init__(self, train_batch_size: int, test_batch_size: int, name: str = 'base_dataloader', *args, **kwargs):
        """
        The initialization method of base dataloader.

        Parameters
        ----------
        name: str, default = 'base_dataloader'
            The name of the base loader class object.

        Returns
        ----------
        object
            The initialized object of the base dataloader class.
        """
        self.name = name
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    @staticmethod
    def from_config(configs: dict):
        if configs is None:
            raise ValueError("configs cannot be None")
        assert 'data_class' in configs
        class_name = configs['data_class']
        parameters = configs['data_parameters'] if 'data_parameters' in configs else {}
        return config.get_obj_from_str(class_name)(**parameters)

    def to_config(self):
        class_name = self.__class__.__name__
        attributes = {attr: getattr(self, attr) for attr in self.__dict__}

        return {
            "data_class": class_name,
            "data_parameters": attributes
        }

    @staticmethod
    def encode_str_labels(labels: Union[List, Tuple, np.array], one_hot: bool = False, device: str = 'cpu'):
        if labels is None or len(labels) == 0:
            raise ValueError("labels cannot be None")

        classes = set(labels)
        if one_hot:
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        else:
            classes_dict = {c: i for i, c in enumerate(classes)}
        encoded_labels = np.array(list(map(classes_dict.get, labels)))
        labels_onehot = torch.tensor(encoded_labels, dtype=torch.long, device=device)
        return labels_onehot

    @abstractmethod
    def load(self, *args, **kwargs):
        """
        The load function of the base dataloader class.

        It loads the data from file in the dataloader class.
        This method is declared to be abstract, and needs to be implemented in the inherited class.

        """
        pass

import time

class dataset(Dataset):
    """
    The dataset base class.

    It defines the template of the dataset, composed of X, y and optional encoder of the features.

    Attributes
    ----------
    X: Any
        The inputs/features of the data instances in the batch.
    y: Any
        The outputs/labels of the data instances in the batch.
    encoder: Any, default = None
        The optional encoder, which can be used for text dataset to project text to the embeddings.

    Methods
    ----------
    __init__
        The dataset initialization method.

    __len__
        The size method of the input data batch.

    __getitem__
        The item retrieval method of the input data batch with certain index.
    """
    def __init__(self, X, y, encoder=None, *args, **kwargs):
        """
        The initialization method of the base dataset class.

        It initializes the dataset class object,
        involving the input features X, output labels y and the optional encoder.

        Parameters
        ----------
        X: Any
            The inputs/features of the data instances in the batch.
        y: Any
            The outputs/labels of the data instances in the batch.
        encoder: Any, default = None
            The optional encoder, which can be used for text dataset to project text to the embeddings.

        Returns
        ----------
        object
            The initialized object of the base dataset.
        """
        super().__init__()
        self.X = X
        self.y = y
        self.encoder = encoder

    def __len__(self):
        """
        The batch size method.

        It reimplements the built-in batch size method.

        Returns
        -------
        int
            The batch size of the input data instance set.
        """
        return len(self.X)

    def __getitem__(self, idx, *args, **kwargs):
        """
        The item retrieval method.

        It returns the feature and label of data instances with certain index.

        Parameters
        ----------
        idx: int
            The index of the data instance to be retrieved.

        Returns
        -------
        tuple
            The retrieved feature and label of the data instance.
        """
        if self.encoder is None:
            sample = self.X[idx]
            target = self.y[idx]
            return sample, target
        else:
            # for the text dataset, the encoder will be applied to obtain its embeddings
            sample = self.encoder(torch.unsqueeze(self.X[idx], 0))
            target = self.y[idx]
            return torch.squeeze(sample, dim=0), target
