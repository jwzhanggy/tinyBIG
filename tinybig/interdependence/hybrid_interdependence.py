# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Hybrid Interdependence #
##########################

from typing import Callable

import torch

from tinybig.interdependence import interdependence
from tinybig.config.base_config import config


class hybrid_interdependence(interdependence):

    def __init__(
        self,
        b: int, m: int,
        interdependence_type: str = 'attribute',
        name: str = 'hybrid_interdependence',
        interdependence_functions: list = None,
        interdependence_function_configs: list = None,
        fusion_function: Callable = None,
        fusion_function_config: dict = None,
        device: str = 'cpu',
        *args, **kwargs
    ):
        if interdependence_functions is not None:
            self.interdependence_functions = interdependence_functions
            for func in self.interdependence_functions:
                func.b = b
                func.m = m
                func.interdependence_type = interdependence_type
        elif interdependence_function_configs is not None:
            self.interdependence_functions = []
            for function_config in interdependence_function_configs:
                assert 'function_class' in function_config
                function_class = function_config['function_class']
                function_parameters = function_config['function_parameters'] if 'function_parameters' in function_config else {}
                function_parameters['b'] = b
                function_parameters['m'] = m
                function_parameters['interdependence_type'] = interdependence_type
                self.interdependence_functions.append(config.get_obj_from_str(function_class)(**function_parameters))
        else:
            raise ValueError('No interdependence functions or configurations are specified...')

        require_parameters = any([func.require_parameters for func in self.interdependence_functions])
        require_data = any([func.require_data for func in self.interdependence_functions])
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_parameters=require_parameters, require_data=require_data, device=device, *args, **kwargs)

        if fusion_function is not None:
            self.fusion_function = fusion_function
        elif fusion_function_config is not None:
            assert 'function_class' in fusion_function_config
            function_class = fusion_function_config['function_class']
            function_parameters = fusion_function_config['function_parameters'] if 'function_parameters' in fusion_function_config else {}
            self.fusion_function = config.get_obj_from_str(function_class)(**function_parameters)
        else:
            raise ValueError('No fusion function or configurations are specified...')

    def get_function_parameter_numbers(self):
        l_list = []
        for func in self.interdependence_functions:
            if func.require_parameters:
                l_list.append(func.calculate_l())
            else:
                l_list.append(0)
        if self.fusion_function.require_parameters:
            l_list.append(self.fusion_function.calculate_l())
        else:
            l_list.append(0)
        return l_list

    def calculate_l(self):
        l_list = self.get_function_parameter_numbers()
        return sum(l_list)

    def calculate_b_prime(self, b: int = None):
        b_prime_list = [func.calculate_b_prime(b=b) for func in self.interdependence_functions]
        return self.fusion_function.calculate_n(dims=b_prime_list)

    def calculate_m_prime(self, m: int = None):
        m_prime_list = [func.calculate_b_prime(m=m) for func in self.interdependence_functions]
        return self.fusion_function.calculate_n(dims=m_prime_list)

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            parameter_numbers = self.get_function_parameter_numbers()
            assert w.numel() == sum(parameter_numbers)

            x = self.pre_process(x=x, device=device)

            A_list = []
            sparse_tag = False
            if w is not None:
                w_segments = torch.split(w, parameter_numbers, dim=-1)
            else:
                w_segments = [None] * len(parameter_numbers)

            for func, w_segment in zip(self.interdependence_functions, w_segments):
                A = func.calculate_A(x=x, w=w_segment, device=device, *args, **kwargs)
                if A.is_sparse:
                    A = A.to_dense()
                    sparse_tag = True
                A_list.append(A)

            if self.fusion_function.require_parameters:
                A = self.fusion_function(x=A_list, w=w_segments[-1], device=device, *args, **kwargs)
            else:
                A = self.fusion_function(x=A_list, device=device, *args, **kwargs)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())
            if sparse_tag:
                A = A.to_sparse_coo()

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A

            return A


