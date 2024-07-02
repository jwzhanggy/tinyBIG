# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Base fabrication function for parameters.

This module contains the base parameter fabrication function class definition.
The parameter reconciliation functions are all defined based on this transformation class.
"""

from abc import abstractmethod
import torch

##############################
# Parameter Fabrication Base #
##############################


class fabrication(torch.nn.Module):
    """
    The base class of the parameter fabrication function in the tinyBIG toolkit.

    It will be used as the base class template for defining the parameter reconciliation functions.

    ...

    Attributes
    ----------
    name: str, default = 'base_fabrication'
        Name of the parameter fabrication function.
    require_parameters: bool, default = True
        Boolean tag of whether the function requires parameters.
    enable_bias: bool, default = False
        Boolean tag of whether the bias is enabled or not.
    device: str, default = 'cpu'
        Device of the parameter fabrication function.

    Methods
    ----------
    __init__
        It initializes the parameter fabrication function.

    get_name
        It gets the name of the parameter fabrication function.

    calculate_l
        It calculates the length of required parameters.

    __call__
        It reimplementation the build-in callable method of the parameter fabrication.
    """
    def __init__(
            self,
            name: str = 'base_fabrication',
            require_parameters: bool = True,
            enable_bias: bool = False,
            device: str = 'cpu',
            *args, **kwargs
    ):
        """
        The initialization method of the base parameter fabrication function.

        It initializes a base parameter fabrication function object.

        Parameters
        ----------
        name: str, default = 'base_fabrication'
            Name of the parameter fabrication function.
        require_parameters: bool, default = True
            Boolean tag of whether the function requires parameters.
        enable_bias: bool, default = False
            Boolean tag of whether the bias is enabled or not.
        device: str, default = 'cpu'
            The device of the parameter fabrication function.

        Returns
        ----------
        object
            The parameter fabrication function object.
        """
        super().__init__()
        self.name = name
        self.device = device
        self.require_parameters = require_parameters
        self.enable_bias = enable_bias

    def get_name(self):
        """
        The name retrieval method of the parameter fabrication function.

        It returns the name of the parameter fabrication function.

        Returns
        -------
        str
            The name of the parameter fabrication function.
        """
        return self.name

    @abstractmethod
    def calculate_l(self, n: int, D: int):
        """
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., l, of the parameter reconciliation function
        based on the intermediate and output space dimensions, n and D.
        The method is declared as an abstractmethod and needs to be implemented in the inherited classes.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.

        Returns
        -------
        int
            The number of required learnable parameters l.
        """
        pass

    @abstractmethod
    def __call__(self, n: int, D: int, w: torch.nn.Parameter, *args, **kwargs):
        """
        The callable method of the parameter reconciliation function.

        It applies the parameter reconciliation operation to the input parameter of length l,
        and returns the reconciled parameter matrix of shape (n, D).
        The method is declared as an abstractmethod and needs to be implemented in the inherited classes.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.
        w: torch.nn.Parameter
            The learnable parameters of the model of length l.

        Returns
        ----------
        torch.Tensor
            The reconciled parameter matrix of shape (n, D).
        """
        pass