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

    This class defines a base structure for loading data from files and provides utility methods
    for configuration management and label encoding.

    Attributes
    ----------
    name: str
        The name of the dataloader instance.
    train_batch_size: int
        The batch size for training data.
    test_batch_size: int
        The batch size for testing data.

    Methods
    -------
    __init__(train_batch_size: int, test_batch_size: int, name: str = 'base_dataloader', *args, **kwargs)
        Initializes the base dataloader class.
    from_config(configs: dict)
        Instantiates a dataloader object from a configuration dictionary.
    to_config()
        Exports the dataloader object to a configuration dictionary.
    encode_str_labels(labels: Union[List, Tuple, np.array], one_hot: bool = False, device: str = 'cpu')
        Encodes string labels into numeric representations, optionally as one-hot vectors.
    load(*args, **kwargs)
        Abstract method to load data from a file, to be implemented by subclasses.
    """
    def __init__(self, train_batch_size: int, test_batch_size: int, name: str = 'base_dataloader', *args, **kwargs):
        """
        Initializes the base dataloader class.

        Parameters
        ----------
        train_batch_size: int
            The batch size for training data.
        test_batch_size: int
            The batch size for testing data.
        name: str, default = 'base_dataloader'
            The name of the dataloader instance.

        Returns
        -------
        None
        """
        self.name = name
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    @staticmethod
    def from_config(configs: dict):
        """
        Instantiates a dataloader object from a configuration dictionary.

        Parameters
        ----------
        configs: dict
            The configuration dictionary containing 'data_class' and optional 'data_parameters'.

        Returns
        -------
        dataloader
            An instance of the dataloader class specified in the configuration.

        Raises
        ------
        ValueError
            If the provided configuration is None or lacks the 'data_class' key.
        """
        if configs is None:
            raise ValueError("configs cannot be None")
        assert 'data_class' in configs
        class_name = configs['data_class']
        parameters = configs['data_parameters'] if 'data_parameters' in configs else {}
        return config.get_obj_from_str(class_name)(**parameters)

    def to_config(self):
        """
        Exports the dataloader object to a configuration dictionary.

        Returns
        -------
        dict
            A dictionary containing the class name and attributes of the dataloader instance.
        """
        class_name = self.__class__.__name__
        attributes = {attr: getattr(self, attr) for attr in self.__dict__}

        return {
            "data_class": class_name,
            "data_parameters": attributes
        }

    @staticmethod
    def encode_str_labels(labels: Union[List, Tuple, np.array], one_hot: bool = False, device: str = 'cpu'):
        """
        Encodes string labels into numeric representations.

        Parameters
        ----------
        labels: Union[List, Tuple, np.array]
            The list of string labels to encode.
        one_hot: bool, default = False
            Whether to encode labels as one-hot vectors.
        device: str, default = 'cpu'
            The device to use for the encoded tensor.

        Returns
        -------
        torch.Tensor
            Encoded labels as a tensor.

        Raises
        ------
        ValueError
            If the labels are None or empty.
        """
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
        Abstract method for loading data from a file.

        This method must be implemented in subclasses to define specific data loading logic.

        Returns
        -------
        None
        """
        pass


class dataset(Dataset):
    """
    The dataset base class.

    This class serves as a template for datasets, providing basic functionality for
    managing input features, labels, and optional feature encoders.

    Attributes
    ----------
    X: Any
        The input features of the dataset instances.
    y: Any
        The output labels of the dataset instances.
    encoder: Any, default = None
        An optional encoder for feature transformations (e.g., text to embeddings).

    Methods
    -------
    __init__(X, y, encoder=None, *args, **kwargs)
        Initializes the dataset with input features, labels, and an optional encoder.
    __len__()
        Returns the number of instances in the dataset.
    __getitem__(idx, *args, **kwargs)
        Retrieves the feature and label of a data instance by index.
    """
    def __init__(self, X, y, encoder=None, *args, **kwargs):
        """
        Initializes the dataset class.

        Parameters
        ----------
        X: Any
            The input features of the dataset instances.
        y: Any
            The output labels of the dataset instances.
        encoder: Any, default = None
            An optional encoder for feature transformations (e.g., text to embeddings).

        Returns
        -------
        None
        """
        super().__init__()
        self.X = X
        self.y = y
        self.encoder = encoder

    def __len__(self):
        """
        Returns the number of instances in the dataset.

        Returns
        -------
        int
            The size of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx, *args, **kwargs):
        """
        Retrieves the feature and label of a data instance by index.

        If an encoder is defined, the feature is transformed using the encoder.

        Parameters
        ----------
        idx: int
            The index of the data instance to retrieve.

        Returns
        -------
        tuple
            A tuple containing the feature and label of the data instance.
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
