# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################
# Remainder Base #
##################

"""
Base remainder function.

This module contains the base remainder function class definition.
The other remainder functions included in the remainder directory are all defined based on this base class.
"""

from abc import abstractmethod
import torch

from torch.nn import Module

from tinybig.module.base_function import function
from tinybig.config import config


class remainder(Module, function):
    r"""
    The base class of the remainder function in the tinyBIG toolkit.

    It will be used as the base class template for defining the remainder functions.

    ...

    Notes
    ----------
    Formally, to approximate the underlying data distribution mapping $f: {R}^m \to {R}^n$ to be learned,
    in addition to the data expansion function and parameter reconciliation function, the remainder function
    $\pi$ completes the approximation as a residual term, governing the learning completeness of the RPN model,
    which can be represented as follows

    $$ \pi: {R}^m \to {R}^{n}.$$

    Without specific descriptions, the remainder function $\pi$ defined here is based solely on the input data $\mathbf{x}$.
    However, in practice, we also allow $\pi$ to include learnable parameters for output dimension adjustment.
    In such cases, it should be rewritten as $\pi(\mathbf{x} | \mathbf{w}')$, where $\mathbf{w}'$ is one extra fraction of the
    model's learnable parameters.

    Attributes
    ----------
    name: str, default = 'base_remainder'
        Name of the remainder function.
    require_remainder_parameters: bool, default = False
        Boolean tag of whether the function requires parameters.
    enable_remainder_bias: bool, default = False
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

    activation
        It applies the activation functions to data calculated in this remainder function.

    forward
        The forward method to calculate the remainder term.

    __call__
        The build-in callable method of the remainder function.
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
        Module.__init__(self)
        function.__init__(self, name=name, device=device)

        self.require_parameters = require_parameters
        self.enable_bias = enable_bias
        self.activation_functions = config.instantiation_functions(activation_functions, activation_function_configs, device=self.device)

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
        return function.func_x(x, self.activation_functions, device=device)

    def to_config(self):
        """
        Converts the current instance of the `remainder` function into a configuration dictionary.

        This method creates a configuration dictionary containing the class name and the current
        attributes of the remainder function. The `activation_functions` attribute is excluded from the
        attributes dictionary and replaced with its configuration details if it is defined.

        Returns
        -------
        dict
            A dictionary representing the configuration of the instance with the following structure:
            {
                "function_class": str,
                    The fully qualified class name, including the module and class name.
                "function_parameters": dict,
                    The instance attributes as key-value pairs, excluding `activation_functions`.
                    If `activation_functions` is defined, its configuration is included under
                    `activation_function_configs`.
            }

        See Also
        --------
        function.functions_to_configs : Converts functions or a list of functions into configuration dictionaries.
        """
        class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        attributes = {attr: getattr(self, attr) for attr in self.__dict__}
        attributes.pop('activation_functions')

        if self.activation_functions is not None:
            attributes['activation_function_configs'] = function.functions_to_configs(self.activation_functions)

        return {
            "function_class": class_name,
            "function_parameters": attributes
        }

    def __call__(self, *args, **kwargs):
        """
        The re-implementation of the callable method.

        It calculates the remainder term based on the inputs. For some remainder functions, this method
        will also accept parameters as the input, which can be applied to the input data for remainder calculation.
        This method will execute by calling the "forward" method.

        Returns
        ----------
        torch.Tensor
            The remainder term calculated based on the input.
        """
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        The forward method of the remainder function.

        It calculates the remainder term based on the inputs. For some remainder functions, this method
        will also accept parameters as the input, which can be applied to the input data for remainder calculation.
        The method is declared as an abstractmethod and needs to be implemented in the inherited classes.

        Returns
        ----------
        torch.Tensor
            The remainder term calculated based on the input.
        """
        pass