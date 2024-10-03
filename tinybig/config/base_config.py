# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
The base config class.

This module implements the base config class that can be used for the RPN models.
The base config class can be used as the config template for other RPN based models as well.
"""
import abc

import yaml
import json
import warnings
import importlib
from typing import Callable
from abc import abstractmethod


class config:
    """
    The base config class.

    It implements the base config class template, which can be used to represent
    the model and object configurations in the tinyBIG toolkit.

    Attributes
    ----------
    name: str, default = 'base_config'
        Name of the base config.

    Methods
    ----------
    __init__
        The initialization method of base config class.

    load_yaml
        Configuration loading method from yaml file

    save_yaml
        Configuration saving method to yaml file

    load_json
        Configuration loading method from json file

    save_json
        Configuration saving method to json file

    instantiate_model_from_config
        Model instantiation method from configurations

    extract_config_from_model
        Configurations extraction methond from models

    """
    def __init__(self, configs: dict | list = None, name='base_config', device: str = 'cpu', *args, **kwargs):
        """
        The initialization method of base config class.

        It initializes a base config object based on the provided parameters.

        Parameters
        ----------
        name: str, default = 'base_config'
            Name of the base config object.

        Returns
        ----------
        object
            The base config object.
        """
        self.configs = configs
        self.name = name
        self.device = device

    def __repr__(self):
        return self.configs

    def to_dict(self):
        self.get_configs()

    def get_configs(self):
        return self.configs

    def set_configs(self, configs: dict):
        self.configs = configs
        return True

    def get_config_entry(self, key: str):
        if key in self.configs:
            return self.configs[key]
        else:
            return None

    def set_config_entry(self, key: str, value):
        if key in self.configs:
            self.configs[key] = value
            return True
        else:
            return False

    def load_yaml(self, cache_dir='./configs', config_file='config.yaml'):
        """
        Model configuration loading method from yaml file

        It loads the model configurations from a yaml file.

        Parameters
        ----------
        cache_dir: str, default = './configs'
            Directory of the configuration yaml file.
        config_file: str, default = 'config.yaml'
            Name of the configuration yaml file

        Returns
        -------
        dict
            The detailed configurations loaded from the yaml file.
        """
        with open('{}/{}'.format(cache_dir, config_file), 'r') as f:
            self.configs = yaml.safe_load(f)
        return self.configs

    def save_yaml(self, cache_dir='./configs', config_file='config.yaml'):
        """
        Model configuration saving method to yaml file

        It saves the model configurations to a yaml file.

        Parameters
        ----------
        configs: dict
            The model configuration in the dictionary data structure
        cache_dir: str, default = './configs'
            Directory of the configuration yaml file.
        config_file: str, default = 'config.yaml'
            Name of the configuration yaml file

        Returns
        -------
        None
            This method doesn't have the returned values.
        """
        with open('{}/{}'.format(cache_dir, config_file), 'w') as f:
            yaml.dump(self.configs, f)
        return True

    def load_json(self, cache_dir='./configs', config_file='configs.json'):
        """
        Model configuration loading method from json file

        It loads the model configurations from a json file.

        Parameters
        ----------
        cache_dir: str, default = './configs'
            Directory of the configuration json file.
        config_file: str, default = 'configs.json'
            Name of the configuration json file

        Returns
        -------
        dict
            The detailed configurations loaded from the json file.
        """
        with open('{}/{}'.format(cache_dir, config_file), 'r') as f:
            self.configs = json.load(f)
        return self.configs

    def save_json(self, cache_dir='./configs', config_file='configs.json'):
        """
        Model configuration saving method to json file

        It saves the model configurations to a json file.

        Parameters
        ----------
        configs: dict
            The model configuration in the dictionary data structure
        cache_dir: str, default = './configs'
            Directory of the configuration json file.
        config_file: str, default = 'configs.json'
            Name of the configuration json file

        Returns
        -------
        None
            This method doesn't have the returned values.
        """
        with open('{}/{}'.format(cache_dir, config_file), 'w') as f:
            json.dump(self.configs, f)
        return True

    @staticmethod
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

    @staticmethod
    def special_function_processing(func_class, func_parameters, device='cpu', *args, **kwargs):
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
            if 'device' not in func_parameters:
                func_parameters['device'] = device
        return func_class, func_parameters

    @staticmethod
    def instantiation_from_configs(
        configs: dict,
        class_name: str = 'function_class',
        parameter_name: str = 'function_parameters',
        device: str = 'cpu',
        *args, **kwargs
    ):
        instantiation_class = configs[class_name]
        if parameter_name in configs:
            instantiation_parameters = configs[parameter_name]
        else:
            instantiation_parameters = {}
        # some special functions may require the device as a parameter, e.g., 'torch.nn.BatchNorm1d'.
        instantiation_class, instantiation_parameters = config.special_function_processing(instantiation_class, instantiation_parameters, device=device)
        return config.get_obj_from_str(instantiation_class)(**instantiation_parameters)

    @staticmethod
    def instantiation_functions(
        functions: dict | list | object = None,
        function_configs: dict | list = None,
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Function initiation method.

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
            return functions
        elif function_configs is not None:
            # if input function_configs contains one single function config, and is not a list
            if isinstance(function_configs, dict) and 'function_class' in function_configs:
                if 'device' not in function_configs: function_configs['device'] = device
                return config.instantiation_from_configs(function_configs, device=device)
            elif isinstance(function_configs, list):
                func_list = []
                for func_config in function_configs:
                    func_list.append(config.instantiation_from_configs(func_config, device=device))
                return func_list
            elif isinstance(function_configs, dict) and 'function_class' not in function_configs:
                func_dict = {}
                for func_config_name in function_configs:
                    func_config = function_configs[func_config_name]
                    if 'device' not in func_config: func_config['device'] = device
                    func_dict[func_config_name] = config.instantiation_from_configs(func_config, device=device)
                return func_dict
        else:
            return None


    # this utility function will help process the inputs from config file to the model for initialization
    # for both layers and heads, we allow the users to provide different parameter combinations
    # (1) provide "total number" n, "num_alloc" [1, 2, 1, ..., 1], and a list of "configs" [config1, config2, ..., confign]
    # (2) only provide "total number" n, and a list of "configs" [config1, config2, ..., confign], we will auto complete the num_alloc to be [1, 1, 1, ..., 1]
    # (3) only provide "total number" n, and only one "configs" either in a list "[config1]" or just "config1", we will auto complete the num_alloc to be [n]
    # (4) only provide "num_alloc" [1, 2, 1, 3, ...., 1], and a list of configs [config1, config2, ..., configk], we will auto complete the "total num" to be sum(num_alloc)
    # other cases, we will report value errors
    @staticmethod
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

    @abstractmethod
    def to_instantiation(self, configs: dict):
        pass