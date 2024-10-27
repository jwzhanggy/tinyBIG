# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

########################
# Fusion Function Base #
########################

from abc import abstractmethod

import torch
from torch.nn import Module

from tinybig.module.base_function import function
from tinybig.config import config


class fusion(Module, function):

    def __init__(
        self,
        dims: list[int] | tuple[int] = None,
        name: str = 'base_fusion',
        require_parameters: bool = False,
        preprocess_functions=None,
        postprocess_functions=None,
        preprocess_function_configs=None,
        postprocess_function_configs=None,
        device: str = 'cpu',
        *args, **kwargs
    ):
        Module.__init__(self)
        function.__init__(self, name=name, device=device)

        self.dims = dims
        self.require_parameters = require_parameters

        self.preprocess_functions = config.instantiation_functions(preprocess_functions, preprocess_function_configs, device=self.device)
        self.postprocess_functions = config.instantiation_functions(postprocess_functions, postprocess_function_configs, device=self.device)

    def get_name(self):
        return self.name

    def get_dims(self):
        return self.dims

    def get_num(self):
        if self.dims is not None:
            return len(self.dims)
        else:
            return 0

    def get_dim(self, index: int):
        if self.dims is not None:
            if index is not None and 0 <= index <= len(self.dims):
                return self.dims[index]
            else:
                raise ValueError(f'Index {index} is out of dim list bounds...')
        else:
            return None

    def pre_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return function.func_x(x, self.preprocess_functions, device=device)

    def post_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return function.func_x(x, self.postprocess_functions, device=device)

    def to_config(self):
        class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        attributes = {attr: getattr(self, attr) for attr in self.__dict__}
        attributes.pop('preprocess_functions')
        attributes.pop('postprocess_functions')

        if self.preprocess_functions is not None:
            attributes['preprocess_function_configs'] = function.functions_to_configs(self.preprocess_functions)
        if self.postprocess_functions is not None:
            attributes['postprocess_function_configs'] = function.functions_to_configs(self.postprocess_functions)

        return {
            "function_class": class_name,
            "function_parameters": attributes
        }

    @abstractmethod
    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_l(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor], w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        pass
