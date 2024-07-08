# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import importlib
import sympy as sp
import torch.nn.functional as F
from torch import nn
import warnings
import os

import math
import torch
import random
import numpy as np


def get_obj_from_str(string: str, reload: bool = False):
    """
    The object initiation from strings.

    It will initiate an object according to the class description as a string.

    Parameters
    ----------
    string: str
        The object class description as a string,
        e.g., "tinybig.expansion.bspline_expansion" and "torch.nn.functional.sigmoid"
    reload: bool, default = False
        The module reloading boolean tag.

    Returns
    -------
    object
        The initiated object of the corresponding class described by the input parameter "string".
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def process_function_list(functions: list = None, function_configs: list = None, device: str = 'cpu', *args, **kwargs):
    """
    Function initialization method.

    It initializes the data preprocessing functions, postprocessing functions, output processing functions,
    and activation functions, which are used for data expansion, rpn head, and remainder functions.

    Parameters
    ----------
    functions: list, default = None
        The list of functions.
    function_configs: list, default = None
        The list of function configs.
    device: str, default = 'cpu'
        The device for processing the functions.

    Returns
    -------
    list
        The list of initialized functions from either the functions or function_configs.
    """
    if functions is not None:
        func_list = functions
    elif function_configs is not None:
        func_list = []
        for func_config in function_configs:
            function_class = func_config['function_class']
            if 'function_parameters' in func_config:
                function_parameters = func_config['function_parameters']
            else:
                function_parameters = {}
            # some special functions may require the device as a parameter, e.g., 'torch.nn.BatchNorm1d'.
            function_class, function_parameters = special_function_process(function_class, function_parameters, device=device)
            func_list.append(get_obj_from_str(function_class)(**function_parameters))
    else:
        func_list = None
    return func_list


def special_function_process(func_class, func_parameters, device='cpu', *args, **kwargs):
    """
    Special function processing method.

    It handles some special functions to accommodate their requirements, like the batchnorms.

    Parameters
    ----------
    func_class: str
        The function class information.
    func_parameters: dict
        The dictionary of function parameters.
    device: str, default = 'cpu'
        The device for hosting and processing these special functions.

    Returns
    -------
    tuple | list
        The tuple of processed function class, and function parameters.
    """
    if func_class in ['torch.nn.BatchNorm1d', 'torch.nn.BatchNorm2d', 'torch.nn.BatchNorm3d']:
        func_parameters['device'] = device
    return func_class, func_parameters


def register_function_parameters(model, func_list):
    """
    Function parameter registration method.

    It registers the parameters involved in some data processing functions, e.g., BatchNorm1d.

    Parameters
    ----------
    model: Any
        The rpn model.
    func_list: list
        The list of data processing functions.

    Returns
    -------
    None
        This functon doesn't have any return values.
    """
    if not hasattr(func_list, '__iter__'):
        func_list = [func_list]
    for idx, function in enumerate(func_list):
        if hasattr(function, 'weight') and hasattr(function, 'bias'):
            model.register_parameter(f'layer_{idx}_weight', function.weight)
            model.register_parameter(f'layer_{idx}_bias', function.bias)
            if hasattr(function, 'running_mean') and hasattr(function, 'running_var'):
                model.register_buffer(f'layer_{idx}_running_mean', function.running_mean)
                model.register_buffer(f'layer_{idx}_running_var', function.running_var)



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
                x = str_func_x(x=x, func=f, device=device)
        return x
    else:
        if callable(functions):
            return functions(x)
        elif type(functions) is str:
            return str_func_x(x=x, func=functions, device=device)


def str_func_x(x, func, device='cpu', *args, **kwargs):
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
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
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
                'input function {} not recognized, the original input x will be returned by default...'.format(func),
                UserWarning)
            return x
    #------------------------------
    # All functions from configs will convert from str to object first
    elif type(func) is str:
        try:
            if '.' in func:
                func = get_obj_from_str(func)
            else:
                func = get_obj_from_str("torch.nn.functional.{}".format(func.lower()))
        except:
            raise ValueError(
                'function {} does\'t belong to "torch.nn.functional.", please provide the complete callable function path, such as "torch.nn.functional.sigmoid..."'.format(
                    func))
        return str_func_x(x, func, device=device, *args, **kwargs)
    else:
        warnings.warn('input function not recognized, the original input x will be returned by default...', UserWarning)
        return x


# this utility function will help process the inputs from config file to the model for initialization
# for both layers and heads, we allow the users to provide different parameter combinations
# (1) provide "total number" n, "num_alloc" [1, 2, 1, ..., 1], and a list of "configs" [config1, config2, ..., confign]
# (2) only provide "total number" n, and a list of "configs" [config1, config2, ..., confign], we will auto complete the num_alloc to be [1, 1, 1, ..., 1]
# (3) only provide "total number" n, and only one "configs" either in a list "[config1]" or just "config1", we will auto complete the num_alloc to be [n]
# (4) only provide "num_alloc" [1, 2, 1, 3, ...., 1], and a list of configs [config1, config2, ..., configk], we will auto complete the "total num" to be sum(num_alloc)
# other cases, we will report value errors
def process_num_alloc_configs(num: int = None, num_alloc: int | list = None, configs: dict | list = None):
    """
    Configuration processing method.

    It processes the provided information about the provided configuration information, including the total number,
    allocation of these numbers, and the list of configurations.

    For the RPN layer and RPN model, they may contain multi-head, and multi-layer.
    To provide more flexibility in their initialization, the tinyBIG toolkit allows users to provide the configuration
    information in different ways:

    * provide "total number" n, "num_alloc" [1, 2, 1, ..., 1], and a list of "configs" [config1, config2, ..., confign]
    * only provide "total number" n, and a list of "configs" [config1, config2, ..., confign], we will auto complete the num_alloc to be [1, 1, 1, ..., 1]
    * only provide "total number" n, and only one "configs" either in a list "[config1]" or just "config1", we will auto complete the num_alloc to be [n]
    * only provide "num_alloc" [1, 2, 1, 3, ...., 1], and a list of configs [config1, config2, ..., configk], we will auto complete the "total num" to be sum(num_alloc)
    * other cases, we will report value errors

    Therefore, this method may need to process such provided parameters to figure out the intended configurations of
    the RPN heads and RPN layers.

    Parameters
    ----------
    num: int, default = None
        Total number of the configurations.
    num_alloc: int | list, default = None
        The allocation of the configuration number.
    configs: dict | list, default = None
        The list/dict of the configurations.

    Returns
    -------
    tuple | pair
        The processed num, num_alloc, configs tuple.
    """
    if num_alloc is None:
        if type(configs) is not list:
            configs = [configs]
        if len(configs) == num:
            num_alloc = [1] * num
        else:
            if num is None:
                if configs is None:
                    raise ValueError(
                        "Neither total num, num_alloc or configs has been provided...")
                else:
                    num = len(configs)
                    num_alloc = [1] * len(configs)
                    warnings.warn(
                        "Neither total num or num_alloc is provided, which will be inferred from the config...".format(
                            len(configs)))
            else:
                if len(configs) == 1:
                    # only one config is provided, repeat the identical config for all heads
                    warnings.warn(
                        "The provided total number {} and config number {} are inconsistent, we will repeat the config {} times by default...".format(
                            num, len(configs), num), UserWarning)
                    num_alloc = [num]
                else:
                    # multiple configs provided but the numbers are inconsistent with number, cannot infer the configs for each head
                    raise ValueError(
                        "The provided total number {} and config number {} are inconsistent and num_alloc parameter is None... please also provide the num_alloc as well...".format(
                            num, len(configs)))
    else:
        # check variable consistency
        if type(num_alloc) is not list:
            num_alloc = [num_alloc]
        if type(configs) is not list:
            configs = [configs]
        if num is None:
            num = sum(num_alloc)

    return num, num_alloc, configs


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


def create_directory_if_not_exists(complete_file_path):
    """
    The directory creation method.

    It checks whether the target file directory exists or not,
    if it doesn't exist, this method will create the directory.

    Parameters
    ----------
    complete_file_path: str
        The complete file path (covering the directory and file name) as a string.

    Returns
    -------
    None
        This method doesn't have any return values.
    """
    directory_path = os.path.dirname(complete_file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' doesn't exit, and it was created...")


def set_random_seed(random_seed: int = 0):
    """
    Random seed setup method.

    It sets up the random seeds for the RPN model prior to model training and testing.

    Specifically, this method will set up the random seeds and related configurations of multiple packages,
    including
    * numpy
    * random
    * torch
    * torch.cuda
    * torch.cudnn
    * torch.mps

    Parameters
    ----------
    random_seed: int, default = 0
        The random seed to be setup.

    Returns
    -------
    None
        This method doesn't have any return values.
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(random_seed)
