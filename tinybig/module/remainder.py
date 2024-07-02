# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Base remainder function.

This module contains the base remainder function class definition.
The other remainder functions included in the remainder directory are all defined based on this base class.
"""

from abc import abstractmethod
import torch

from tinybig.util import process_function_list, func_x

##################
# Remainder Base #
##################


class remainder(torch.nn.Module):
    """
    The base class of the remainder function in the tinyBIG toolkit.

    It will be used as the base class template for defining the remainder functions.

    ...

    Attributes
    ----------
    name: str, default = 'base_remainder'
        Name of the remainder function.
    require_parameters: bool, default = False
        Boolean tag of whether the function requires parameters.
    enable_bias: bool, default = False
        Boolean tag of whether the bias is enabled or not.
    activation_functions: list, default = None
        The list of activation functions that can be applied in the remainder function.
    activation_function_configs: list, default = None
        The list of activation function configs that can be applied in the remainder function.
    device: str, default = 'cpu'
        Device of the remainder function.

    Methods
    ----------
    __init__
        It initializes the remainder function.

    get_name
        It gets the name of the remainder function.

    activation
        It applies the activation functions to data calculated in this remainder function.

    __call__
        It reimplementation the build-in callable method of the remainder function.
    """
    def __init__(
            self,
            name='base_remainder',
            require_parameters=False,
            enable_bias=False,
            activation_functions=None,
            activation_function_configs=None,
            device='cpu',
            *args, **kwargs
    ):
        """
        The initialization method of the base remainder function.

        It initializes a base remainder function object.

        Parameters
        ----------
        name: str, default = 'base_remainder'
            Name of the remainder function.
        require_parameters: bool, default = False
            Boolean tag of whether the function requires parameters.
        enable_bias: bool, default = False
            Boolean tag of whether the bias is enabled or not.
        activation_functions: list, default = None
            The list of activation functions that can be applied in the remainder function.
        activation_function_configs: list, default = None
            The list of activation function configs that can be applied in the remainder function.
        device: str, default = 'cpu'
            Device of the remainder function.

        Returns
        ----------
        object
            The remainder function object.
        """
        super().__init__()
        self.name = name
        self.device = device
        self.require_parameters = require_parameters
        self.enable_bias = enable_bias
        self.activation_functions = process_function_list(activation_functions, activation_function_configs, device=self.device)
        #register_function_parameters(self, self.activation_functions)

    def get_name(self):
        """
        The name retrieval method of the remainder function.

        It returns the name of the remainder function.

        Returns
        -------
        str
            The name of the remainder function.
        """

        return self.name

    def activation(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        The activation method of remainder function.

        It processes the remainder term with the (optional) activation functions.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.

        Returns
        -------
        Tensor
            It returns the updated remainder term processed by the activation functions.
        """
        return func_x(x, self.activation_functions, device=device)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        The callable method of the remainder function.

        It calculates the remainder term based on the inputs. For some remainder functions, this method
        will also accept parameters as the input, which can be applied to the input data for remainder calculation.
        The method is declared as an abstractmethod and needs to be implemented in the inherited classes.

        Returns
        ----------
        torch.Tensor
            The remainder term calculated based on the input.
        """
        pass