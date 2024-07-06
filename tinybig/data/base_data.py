# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
The base dataloader and dataset classes.

This module implements the base dataloader class and dataset class,
which can be used for loading datasets for RPN model training and testing in tinyBIG toolkit.
"""
from abc import abstractmethod

import torch
from torch.utils.data import Dataset


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
    def __init__(self, name='base_dataloader', *args, **kwargs):
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

    @abstractmethod
    def load(self, *args, **kwargs):
        """
        The load function of the base dataloader class.

        It loads the data from file in the dataloader class.
        This method is declared to be abstract, and needs to be implemented in the inherited class.

        """
        pass


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
