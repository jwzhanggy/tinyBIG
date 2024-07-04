# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
The base config class.

This module implements the base config class that can be used for the RPN models.
The base config class can be used as the config template for other RPN based models as well.
"""
import yaml
import json
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
    def __init__(self, name='base_config', *args, **kwargs):
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
        self.name = name

    @staticmethod
    def load_yaml(cache_dir='./configs', config_file='config.yaml'):
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
            configs = yaml.safe_load(f)
        return configs

    @staticmethod
    def save_yaml(configs, cache_dir='./configs', config_file='config.yaml'):
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
            yaml.dump(configs, f)

    @staticmethod
    def load_json(cache_dir='./configs', config_file='configs.json'):
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
            configs = json.load(f)
        return configs

    @staticmethod
    def save_json(configs, cache_dir='./configs', config_file='configs.json'):
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
            json.dump(configs, f)

    @abstractmethod
    def instantiate_object_from_config(self, *args, **kwargs):
        """
        Model object instantiation from the configurations.

        It instantiates a model object from the detailed configurations.
        This method is declared as an abstract method, which needs to be implemented in its inherited class.
        """
        pass

    @abstractmethod
    def extract_config_from_object(self, *args, **kwargs):
        """
        Model configuration extraction from the model object.

        It extracts a model's configuration from a model object.
        This method is declared as an abstract method, which needs to be implemented in its inherited class.
        """
        pass

