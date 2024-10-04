# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################
# Base Functions #
##################

import warnings
import sympy as sp
from typing import Callable
from abc import abstractmethod

import torch
import torch.nn.functional as F

from tinybig.config.base_config import config


class function:
    def __init__(self, name: str = 'base_function', device: str = 'cpu', *args, **kwargs):
        self.name = name
        self.device = device

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)

    def to_config(self):
        class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        attributes = {attr: getattr(self, attr) for attr in self.__dict__}
        return {
            "function_class": class_name,
            "function_parameters": attributes
        }

    @staticmethod
    def func_x(x, functions, device: str = 'cpu'):
        """
        The function execution to the input data.

        It applies the list of functions to the input vector and returns the calculation results.

        This method will be extensively called for handeling the data processing functions in the
        expansion functions, RPN head and remainder functions in tinyBIG.

        * preprocess_functions in expansion functions
        * postprocess_functions in expansion functions
        * output_process_functions in rpn heads
        * activation_functions in remainder functions

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        functions: list | tuple | callable
            The functions to be applied to the input vector. The function can be callable functions,
            string names of the functions, the complete class descriptions of the functions, etc.
        device: str, default = 'cpu'
            The device to perform the function on the input vector.

        Returns
        -------
        torch.Tensor
            The processed input vector by these functions.
        """
        if functions is None or ((isinstance(functions, list) or isinstance(functions, tuple)) and len(functions) == 0):
            return x
        elif isinstance(functions, list) or isinstance(functions, tuple):
            for f in functions:
                if callable(f):
                    x = f(x)
                elif type(f) is str:
                    x = function.str_func_x(x=x, func=f, device=device)
            return x
        else:
            if callable(functions):
                return functions(x)
            elif type(functions) is str:
                return function.str_func_x(x=x, func=functions, device=device)

    @staticmethod
    def str_func_x(x, func: str | Callable, device='cpu', *args, **kwargs):
        """
        Function recognition from their string names or string class descriptions.

        It recognizes the data processing functions from their names or class description in strings,
        e.g., "layer_norm" or "torch.nn.functional.layer_norm".

        Since these functions can be very diverse, whose definitions are also very different,
        it makes it very challenging to process them based on their string descriptions.
        This method can process some basic functions, e.g., activation functions, and normalization functions.
        For the functions that are not implemented in this method, the users may consider to extend the method
        to handle more complex input functions.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        func: str
            The string description of the functoin name or class.
        device: str, default = 'cpu'
            The device to host and apply the recognized functions.

        Returns
        -------
        torch.Tensor
            The processed input data vector by the recognized functions.
        """
        if func is None:
            return x
        elif callable(func):
            # --------------------------
            if func in [F.sigmoid, F.relu, F.leaky_relu, F.tanh, F.softplus, F.silu, F.celu, F.gelu]:
                return func(x)
            # --------------------------
            # dropout functions
            elif func in [
                F.dropout
            ]:
                # --------------------------
                if 'p' in kwargs:
                    p = kwargs['p']
                else:
                    p = 0.5
                # --------------------------
                if func in [F.dropout]:
                    return func(x, p=p)
                else:
                    return func(p=p)(x)
            # --------------------------
            # layer_norm functions
            elif func in [F.layer_norm]:
                # --------------------------
                if 'normalized_shape' in kwargs:
                    normalized_shape = kwargs['normalized_shape']
                else:
                    normalized_shape = [x.size(-1)]
                # --------------------------
                if func in [F.layer_norm]:
                    return func(x, normalized_shape=normalized_shape)
                else:
                    return func(normalized_shape=normalized_shape)(x)
                # --------------------------
            # --------------------------
            # batch_norm functions
            elif func in [
                torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                torch.nn.modules.batchnorm.BatchNorm1d,
                torch.nn.modules.batchnorm.BatchNorm2d,
                torch.nn.modules.batchnorm.BatchNorm3d
            ]:
                # --------------------------
                if 'num_features' in kwargs:
                    num_features = kwargs['num_features']
                else:
                    num_features = x.size(-1)
                # ---------------------------
                return func(num_features=num_features, device=device)(x)
            # --------------------------
            # other functions
            elif func in [
                torch.exp,
            ]:
                return func(x)
            # --------------------------
            else:
                warnings.warn(
                    'input function {} not recognized, the original input x will be returned by default...'.format(
                        func),
                    UserWarning)
                return x
        # ------------------------------
        # All functions from configs will convert from str to object first
        elif type(func) is str:
            try:
                if '.' in func:
                    func = config.get_obj_from_str(func)
                else:
                    func = config.get_obj_from_str("torch.nn.functional.{}".format(func.lower()))
            except:
                raise ValueError(
                    'function {} does\'t belong to "torch.nn.functional.", please provide the complete callable function path, such as "torch.nn.functional.sigmoid..."'.format(
                        func))
            return function.str_func_x(x, func, device=device, *args, **kwargs)
        else:
            warnings.warn('input function not recognized, the original input x will be returned by default...',
                          UserWarning)
            return x

    @staticmethod
    def string_to_function(formula, variable):
        """
        Formula recognition from strings.

        It recognizes and returns the formula and variables from strings via the sympy package.

        Parameters
        ----------
        formula: str
            The function formula as a string.
        variable: list
            The list of the variables involved in the formula.

        Returns
        -------
        sympy.FunctionClass
            The recognized function of the input formula.
        """
        # Define the symbol
        var = sp.symbols(variable)

        # Parse the formula string into a sympy expression
        expression = sp.sympify(formula)

        # Convert the sympy expression to a lambda function
        func = sp.lambdify(var, expression, 'numpy')

        return func

    @staticmethod
    def functions_to_configs(functions: list | tuple | Callable, class_name: str = 'function_class', parameter_name: str = 'function_parameters'):
        if functions is None:
            return None
        elif isinstance(functions, Callable):
            func_class_name = f"{functions.__class__.__module__}.{functions.__class__.__name__}"
            func_parameters = {attr: getattr(functions, attr) for attr in functions.__dict__}
            return {
                class_name: func_class_name,
                parameter_name: func_parameters
            }
        else:
            return [
                function.functions_to_configs(func) for func in functions
            ]

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

