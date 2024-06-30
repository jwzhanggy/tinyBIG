# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import yaml
from abc import abstractmethod

from tinybig.util.util import get_obj_from_str


class config:
    def __init__(self, name='base_config', *args, **kwargs):
        self.name = name

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def instantiate_from_config(self, *args, **kwargs):
        pass

    @abstractmethod
    def extract_config_from_input(self, *args, **kwargs):
        pass


class rpn_config(config):

    def __init__(self, name='rpn_detailed_config'):
        super().__init__(name=name)

    def load(self, cache_dir='./configs', config_file='configs.yaml'):
        with open('{}/{}'.format(cache_dir, config_file), 'r') as f:
            configs = yaml.safe_load(f)
        return configs

    def save(self, configs, cache_dir='./configs', config_file='configs.yaml'):
        with open('{}/{}'.format(cache_dir, config_file), 'r') as f:
            yaml.dump(configs, f)

    @staticmethod
    def instantiate_from_config(configs, object_name_list=None):
        object_name_list = object_name_list if object_name_list is not None else ['data', 'model', 'learner', 'metric', 'result']
        object_dict = {}
        for object_name_stem in object_name_list:
            if "{}_class".format(object_name_stem) in configs:
                class_name = configs["{}_class".format(object_name_stem)]
                parameters = configs["{}_parameters".format(object_name_stem)]
                obj = get_obj_from_str(class_name)(**parameters)
            else:
                obj = None
            object_dict[object_name_stem] = obj
        return object_dict

    # TBD, recursively call internal components of input object or object lists
    def extract_config_from_input(self, input_obj, *args, **kwargs):
        if type(input_obj) is list:
            pass
        else:
            pass
