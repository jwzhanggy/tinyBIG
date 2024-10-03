# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################
# RPN Function Config Templates #
#################################

from tinybig.config.base_config import config
from tinybig.util.utility import find_class_in_package


class function_configs(config):
    def __init__(
        self,
        configs: dict | list = None,
        class_name: str = None,
        parameters: dict = None,
        name: str = 'function_configs',
        device: str = 'cpu',
        *args, **kwargs
    ):
        if configs is not None:
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        elif class_name is not None:
            if not class_name.startswith('tinybig'):
                class_name = class_name.split('.')[-1]
                class_name = find_class_in_package(class_name=class_name)
            assert class_name.startswith('tinybig')
            configs = {
                'function_class': class_name,
                'function_parameters': parameters,
            }
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        else:
            super().__init__(name=name, configs=None, device=device, *args, **kwargs)

    def to_instantiation(self):
        if self.configs is None:
            raise ValueError('The function configs object cannot be none...')

        return config.instantiation_from_configs(
            configs=self.configs,
            device=self.device,
            class_name='function_class',
            parameter_name='function_parameters'
        )


class function_list_configs(config):
    def __init__(
        self,
        configs: dict | list = None,
        class_name: list | str = None,
        parameters: list | dict = None,
        name: str = 'function_list_configs',
        device: str = 'cpu',
        *args, **kwargs
    ):
        if configs is not None:
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        elif class_name is not None:
            if isinstance(class_name, str):
                class_name = [class_name]
            if isinstance(parameters, dict):
                parameters = [parameters]
            assert len(class_name) == len(parameters)
            configs = [
                function_configs(
                    class_name=name,
                    parameters=parameter,
                ).get_configs()
                for name, parameter in zip(class_name, parameters)
            ]
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        else:
            super().__init__(name=name, configs=None, device=device, *args, **kwargs)

    def to_instantiation(self):
        if self.configs is None:
            raise ValueError('The function configs object cannot be none...')

        return [config.instantiation_from_configs(
            configs=config,
            device=self.device,
            class_name='function_class',
            parameter_name='function_parameters'
        ) for config in self.configs]



