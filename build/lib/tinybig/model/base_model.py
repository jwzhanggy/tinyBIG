# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
The base model template.

It defines the base model template for implementing the RPN models.
"""

import torch
from abc import abstractmethod

from tinybig.util.util import create_directory_if_not_exists


class model(torch.nn.Module):
    """
    The base model class of the RPN model in the tinyBIG toolkit.

    It inherits from the torch.nn.Module class, which also inherits the
    "state_dict" and "load_state_dict" methods from the base class.

    ...

    Attributes
    ----------
    name: str, default = 'base_metric'
        Name of the model.

    Methods
    ----------
    __init__
        It performs the initialization of the model

    save_ckpt
        It saves the model state as checkpoint to file.

    load_ckpt
        It loads the model state from a file.

    __call__
        It reimplementation the build-in callable method.

    forward
        The forward method of the model.
    """
    def __init__(self, name='model_name', *args, **kwargs):
        """
        The initialization method of the base model class.

        It initializes a model object based on the provided model parameters.

        Parameters
        ----------
        name: str, default = 'model_name'
            The name of the model, with default value "model_name".

        Returns
        ----------
        object
            The initialized model object.
        """
        super().__init__()
        self.name = name

    def save_ckpt(self, cache_dir='./ckpt', checkpoint_file='checkpoint'):
        """
        The model state checkpoint saving method.

        It saves the current model state to a checkpoint file.

        Parameters
        ----------
        cache_dir: str, default = './ckpt'
            The cache directory of the model checkpoint file.
        checkpoint_file: str, default = 'checkpoint'
            The checkpoint file name.

        Returns
        -------
        None
            This method doesn't have return values.
        """
        create_directory_if_not_exists(f'{cache_dir}/{checkpoint_file}')
        torch.save(self.state_dict(), f'{cache_dir}/{checkpoint_file}')
        print("model checkpoint saving to {}/{}...".format(cache_dir, checkpoint_file))

    def load_ckpt(self, cache_dir: str = './ckpt', checkpoint_file: str = 'checkpoint', strict: bool = True):
        """
        The model state checkpoint loading method.

        It loads the model state from the provided checkpoint file.

        Parameters
        ----------
        cache_dir: str, default = './ckpt'
            The cache directory of the model checkpoint file.
        checkpoint_file: str, default = 'checkpoint'
            The checkpoint file name.
        strict: bool, default = True
            The boolean tag of whether the model state loading follows the strict configuration checking.

        Returns
        -------
        None
            This method doesn't have return values.
        """
        self.load_state_dict(torch.load(f'{cache_dir}/{checkpoint_file}'), strict=strict)
        print("model checkpoint loading from {}/{}...".format(cache_dir, checkpoint_file))

    def __call__(self, *args, **kwargs):
        """
        The built-in callable method.

        It will call the forward method to generate the outputs based on the provided inputs.

        Returns
        ----------
        torch.Tensor
            The model generated outputs.
        """
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        The forward method of the model.

        It is declared to be an abstractmethod and needs to be implemented in the inherited RPN model classes.
        This callable method accepts the data instances as the input and generate the desired outputs.

        Returns
        ----------
        torch.Tensor
            The model generated outputs.
        """
        pass