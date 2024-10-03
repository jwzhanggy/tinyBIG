# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

######################################
# Statistical kernel Interdependence #
######################################

import torch
from typing import Callable

from tinybig.interdependence import interdependence
from tinybig.koala.statistics import (
    batch_kl_divergence_kernel,
    batch_pearson_correlation_kernel,
    batch_rv_coefficient_kernel,
    batch_mutual_information_kernel
)


class statistical_kernel_based_interdependence(interdependence):

    def __init__(
        self,
        b: int, m: int, kernel: Callable,
        interdependence_type: str = 'attribute',
        name: str = 'statistical_kernel_based_interdependence',
        require_data: bool = True,
        require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_parameters=require_parameters, require_data=require_data, device=device, *args, **kwargs)

        if kernel is None:
            raise ValueError('the kernel is required for the statistical kernel based interdependence function')
        self.kernel = kernel
        self.kernel = kernel

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            assert x is not None and x.ndim == 2
            x = self.pre_process(x=x, device=device)
            A = self.kernel(x)
            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A

            return A


class kl_divergence_interdependence(statistical_kernel_based_interdependence):
    def __init__(self, name: str = 'kl_divergence_interdependence', *args, **kwargs):
        super().__init__(kernel=batch_kl_divergence_kernel, name=name, *args, **kwargs)


class pearson_correlation_interdependence(statistical_kernel_based_interdependence):
    def __init__(self, name: str = 'pearson_correlation_interdependence', *args, **kwargs):
        super().__init__(kernel=batch_pearson_correlation_kernel, name=name, *args, **kwargs)


class rv_coefficient_interdependence(statistical_kernel_based_interdependence):
    def __init__(self, name: str = 'rv_coefficient_interdependence', *args, **kwargs):
        super().__init__(kernel=batch_rv_coefficient_kernel, name=name, *args, **kwargs)


class mutual_information_interdependence(statistical_kernel_based_interdependence):
    def __init__(self, name: str = 'mutual_information_interdependence', *args, **kwargs):
        super().__init__(kernel=batch_mutual_information_kernel, name=name, *args, **kwargs)


