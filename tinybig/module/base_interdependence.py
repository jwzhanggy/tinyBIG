# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################
# Interdependence Function Base #
#################################

import warnings
from abc import abstractmethod

import torch
from torch.nn import Module

from tinybig.module.base_function import function
from tinybig.config import config


class interdependence(Module, function):

    def __init__(
        self,
        b: int, m: int,
        name: str = 'base_interdependency',
        interdependence_type: str = 'attribute',
        require_data: bool = True,
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

        self.interdependence_type = interdependence_type

        self.b = b
        self.m = m

        self.require_data = require_data
        self.require_parameters = require_parameters

        self.preprocess_functions = config.instantiation_functions(preprocess_functions, preprocess_function_configs, device=self.device)
        self.postprocess_functions = config.instantiation_functions(postprocess_functions, postprocess_function_configs, device=self.device)

        self.A = None

    @property
    def interdependence_type(self):
        return self._interdependence_type

    @interdependence_type.setter
    def interdependence_type(self, value):
        allowed_values = ['instance_interdependence', 'instance', 'left', 'attribute_interdependence', 'attribute', 'right']
        if value not in allowed_values:
            raise ValueError(f"Invalid value for my_string. Allowed values are: {allowed_values}")
        self._interdependence_type = value

    def check_A_shape_validity(self, A: torch.Tensor):
        if A is None:
            raise ValueError("A must be provided")

        assert self.interdependence_type is not None and isinstance(self.interdependence_type, str)

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert self.b is not None
            assert A.shape == (self.b, self.calculate_b_prime(b=self.b))
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert self.m is not None
            assert A.shape == (self.m, self.calculate_m_prime(m=self.m))
        else:
            raise ValueError("The interdependence type {self.interdependence_type} is not supported...}")

    def get_name(self):
        return self.name

    def get_A(self):
        if self.A is None:
            warnings.warn("The A matrix is None...")
            return None
        else:
            return self.A

    def get_b(self):
        return self.b

    def get_m(self):
        return self.m

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

    def calculate_l(self):
        return 0

    def calculate_b_prime(self, b: int = None):
        b = b if b is not None else self.b
        if self.interdependence_type not in ['row', 'left', 'instance', 'instance_interdependence']:
            warnings.warn("The interdependence_type is not about the instances, its b dimension will not be changed...")
        return b

    def calculate_m_prime(self, m: int = None):
        m = m if m is not None else self.m
        if self.interdependence_type not in ['column', 'right', 'attribute', 'attribute_interdependence']:
            warnings.warn("The interdependence_type is not about the attributes, its m dimension will not be changed...")
        return m

    def forward(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, kappa_x: torch.Tensor = None, device: str = 'cpu', *args, **kwargs):
        if self.require_data:
            assert x is not None and x.ndim == 2
        if self.require_parameters:
            assert w is not None and w.ndim == 2

        data_x = kappa_x if kappa_x is not None else x
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            # A shape: b * b'
            A = self.calculate_A(x.transpose(0, 1), w, device=device)
            assert A is not None and A.size(0) == data_x.size(0)
            if data_x.is_sparse or A.is_sparse:
                xi_x = torch.sparse.mm(A.t(), data_x)
            else:
                xi_x = torch.matmul(A.t(), data_x)
            return xi_x
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            # A shape: m * m'
            A = self.calculate_A(x, w, device)
            assert A is not None and A.size(0) == data_x.size(1)
            if data_x.is_sparse or A.is_sparse:
                xi_x = torch.sparse.mm(data_x, A)
            else:
                xi_x = torch.matmul(data_x, A)
            return xi_x
        else:
            raise ValueError(f"Invalid interdependence type: {self.interdependence_type}")


    @abstractmethod
    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        pass

