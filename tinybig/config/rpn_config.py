# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
The RPN model config class.

This module implements the model config class for the RPN model.
the rpn_config inherits from the base config class, and needs to re-implement the
"instantiate_model_from_config" and "extract_config_from_model" methods.
"""

from tinybig.config.base_config import config
from tinybig.util.util import get_obj_from_str


class rpn_config(config):
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
    def __init__(self, name='rpn_config'):
        """
        The rpn config initialization method.

        It initializes the rpn config object based on the provided parameters.

        Parameters
        ----------
        name: str, default = 'rpn_config'
            Name of the rpn configuration object.
        """
        super().__init__(name=name)

    @staticmethod
    def instantiate_object_from_config(configs: dict):
        """
        The rpn object instantiation method from the configuration.

        It initializes a rpn object from its detailed configuration information.

        Parameters
        ----------
        configs: dict
            The rpn object detailed configuration.

        Returns
        -------
        dict
            The initialized object dictionary based on the configs.
        """
        object_dict = {}
        for config in configs:
            if '_configs' not in config: continue
            object_name_stem = config.split('_configs')[0]
            if "{}_class".format(object_name_stem) in configs[config]:
                class_name = configs[config]["{}_class".format(object_name_stem)]
                parameters = configs[config]["{}_parameters".format(object_name_stem)]
                obj = get_obj_from_str(class_name)(**parameters)
            else:
                obj = None
            object_dict[object_name_stem] = obj
        return object_dict

    @staticmethod
    def extract_config_from_object(objects, *args, **kwargs):
        """
        The rpn configuration extraction method from the rpn objects.

        It extracts the rpn object configuration and return it as a confi dictionary.

        Parameters
        ----------
        objects: object
            The rpn model and component object.

        Returns
        -------
        dict
            The configuration information of the input object.
        """
        pass
