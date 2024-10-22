# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################################
# Numerical kernel Interdependence #
####################################

import functools
from typing import Union, Any, Callable, List
import numpy as np

import torch

from tinybig.interdependence import interdependence

from tinybig.koala.linear_algebra import (
    linear_kernel,
    polynomial_kernel,
    hyperbolic_tangent_kernel,
    exponential_kernel,
    cosine_similarity_kernel,
    minkowski_distance_kernel,
    manhattan_distance_kernel,
    euclidean_distance_kernel,
    chebyshev_distance_kernel,
    canberra_distance_kernel,
    gaussian_rbf_kernel,
    laplacian_kernel,
    anisotropic_rbf_kernel,
    custom_hybrid_kernel,
)


class numerical_kernel_based_interdependence(interdependence):

    def __init__(
        self,
        b: int, m: int, kernel: Callable,
        interdependence_type: str = 'attribute',
        name: str = 'kernel_based_interdependence',
        require_data: bool = True,
        require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_parameters=require_parameters, require_data=require_data, device=device, *args, **kwargs)

        if kernel is None:
            raise ValueError('the kernel is required for the kernel based interdependence function')
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
                print(x.shape, A.shape, self.m, self.calculate_m_prime())
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A

            return A


class linear_kernel_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'linear_kernel_interdependence', *args, **kwargs):
        super().__init__(kernel=linear_kernel, name=name, *args, **kwargs)


class cosine_similarity_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'cosine_similarity_interdependence', *args, **kwargs):
        super().__init__(kernel=cosine_similarity_kernel, name=name, *args, **kwargs)


class minkowski_distance_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, p: Union[int, float, str, Any], name: str = 'minkowski_distance_interdependence', *args, **kwargs):
        minkowski_kernel_func = functools.partial(minkowski_distance_kernel, p=p)
        super().__init__(kernel=minkowski_kernel_func, name=name, *args, **kwargs)


class manhattan_distance_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'manhattan_distance_interdependence', *args, **kwargs):
        super().__init__(kernel=manhattan_distance_kernel, name=name, *args, **kwargs)


class euclidean_distance_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'euclidean_distance_interdependence', *args, **kwargs):
        super().__init__(kernel=euclidean_distance_kernel, name=name, *args, **kwargs)


class chebyshev_distance_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'chebyshev_distance_interdependence', *args, **kwargs):
        super().__init__(kernel=chebyshev_distance_kernel, name=name, *args, **kwargs)


class canberra_distance_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'canberra_distance_interdependence', *args, **kwargs):
        super().__init__(kernel=canberra_distance_kernel, name=name, *args, **kwargs)


class polynomial_kernel_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'polynomial_kernel_interdependence', c: float = 0.0, d: int = 1, *args, **kwargs):
        polynomial_kernel_func = functools.partial(polynomial_kernel, c=c, d=d)
        super().__init__(kernel=polynomial_kernel_func, name=name, *args, **kwargs)


class hyperbolic_tangent_kernel_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'hyperbolic_tangent_kernel_interdependence', c: float = 0.0, alpha: float = 1.0, *args, **kwargs):
        hyperbolic_tangent_kernel_func = functools.partial(hyperbolic_tangent_kernel, c=c, alpha=alpha)
        super().__init__(kernel=hyperbolic_tangent_kernel_func, name=name, *args, **kwargs)


class exponential_kernel_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'exponential_kernel_interdependence', gamma: float = 1.0, *args, **kwargs):
        exponential_kernel_func = functools.partial(exponential_kernel, gamma=gamma)
        super().__init__(kernel=exponential_kernel_func, name=name, *args, **kwargs)


class gaussian_rbf_kernel_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'gaussian_rbf_kernel_interdependence', sigma: float = 1.0, *args, **kwargs):
        gaussian_rbf_kernel_func = functools.partial(gaussian_rbf_kernel, sigma=sigma)
        super().__init__(kernel=gaussian_rbf_kernel_func, name=name, *args, **kwargs)


class laplacian_kernel_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'laplacian_kernel_interdependence', sigma: float = 1.0, *args, **kwargs):
        laplacian_kernel_func = functools.partial(laplacian_kernel, sigma=sigma)
        super().__init__(kernel=laplacian_kernel_func, name=name, *args, **kwargs)


class anisotropic_rbf_kernel_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, name: str = 'anisotropic_rbf_kernel_interdependence', a_vector: Union[torch.Tensor, np.array] = None, a_scalar: float = 1.0, *args, **kwargs):
        anisotropic_rbf_kernel_func = functools.partial(anisotropic_rbf_kernel, a_vector=a_vector, a_scalar=a_scalar)
        super().__init__(kernel=anisotropic_rbf_kernel_func, name=name, *args, **kwargs)


class custom_hybrid_kernel_interdependence(numerical_kernel_based_interdependence):
    def __init__(self, kernels: List[Callable[[np.matrix], np.matrix]], weights: List[float] = None, name: str = 'custom_hybrid_kernel_interdependence', *args, **kwargs):
        custom_hybrid_kernel_func = functools.partial(custom_hybrid_kernel, kernels=kernels, weights=weights)
        super().__init__(kernel=custom_hybrid_kernel_func, name=name, *args, **kwargs)

