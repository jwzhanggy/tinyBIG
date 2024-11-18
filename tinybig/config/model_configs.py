# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
# RPN Model Configs #
#####################

"""
The RPN model config class.

This module implements the model config class for the RPN model.
the rpn_config inherits from the base config class, and needs to re-implement the
"instantiate_model_from_config" and "extract_config_from_model" methods.
"""

from tinybig.config.base_config import config


class model_configs(config):
    """
    The configuration class of RPN model.

    It implements the configuration class for the RPN model specifically.
    The class can help instantiate the RPN model from its configuration file.

    Attributes
    ----------
    name: str, default = 'rpn_config'
        Name of the rpn_config object.

    Methods
    ----------

    __init__
        The rpn_config initialization method.

    instantiate_object_from_config
        The rpn objects instantiation method from the configuration.

    extract_config_from_object
        The rpn configuration extraction method from the rpn objects.
    """
    def __init__(
        self,
        configs: dict = None,
        parameters: dict = None,
        name='rpn_config',
        layer_configs: dict | list = None,
        depth: int = None,
        depth_alloc: int | list = None,
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
        The rpn config initialization method.

        It initializes the rpn config object based on the provided parameters.

        Parameters
        ----------
        name: str, default = 'rpn_config'
            Name of the rpn configuration object.
        """

        if configs is not None:
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        elif parameters is not None:
            configs = {
                'model_class': 'tinybig.mode.rpn_model',
                'model_parameters': parameters,
            }
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        elif layer_configs is not None:
            parameters = {
                'name': name,
                'depth': depth,
                'depth_alloc': depth_alloc,
                'layer_configs': layer_configs,
                'device': device,
            }
            configs = {
                'model_class': 'tinybig.mode.rpn_model',
                'model_parameters': parameters,
            }
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        else:
            super().__init__(name=name, configs=None, device=device, *args, **kwargs)

    def to_instantiation(self):
        if self.configs is None:
            raise ValueError('The model configs object cannot be none...')
        self.process_depth_alloc_layer_configs()

        return config.instantiation_from_configs(
            configs=self.configs,
            device=self.device,
            class_name='model_class',
            parameter_name='model_parameters'
        )

    def process_depth_alloc_layer_configs(self):
        if self.configs is None or 'model_class' not in self.configs or 'model_parameters' not in self.configs:
            raise ValueError("the config dict cannot be none and should contain model configuration details...")

        depth = self.configs['model_parameters']['depth'] if 'depth' in self.configs['model_parameters'] else None
        depth_alloc = self.configs['model_parameters']['depth_alloc'] if 'depth_alloc' in self.configs['model_parameters'] else None
        layer_configs = self.configs['model_parameters']['layer_configs'] if 'layer_configs' in self.configs['model_parameters'] else None

        depth, depth_alloc, layer_configs = config.process_num_alloc_configs(
            num=depth,
            num_alloc=depth_alloc,
            configs=layer_configs,
        )

        self.configs['model_parameters']['depth'] = depth
        self.configs['model_parameters']['depth_alloc'] = depth_alloc
        self.configs['model_parameters']['layer_configs'] = layer_configs
