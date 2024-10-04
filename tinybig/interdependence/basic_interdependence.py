# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
import warnings

#########################
# Basic Interdependence #
#########################

import torch

from tinybig.interdependence import interdependence


class constant_interdependence(interdependence):

    def __init__(
        self,
        b: int, m: int,
        A: torch.Tensor,
        interdependence_type: str = 'attribute',
        name: str = 'constant_interdependence',
        device: str = 'cpu',
        *args, **kwargs
    ):
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=False, require_parameters=False, device=device, *args, **kwargs)
        if A is None or A.ndim != 2:
            raise ValueError('The parameter matrix A is required and should have ndim: 2 by default')
        self.A = A
        if self.A.device != device:
            self.A.to(device)

    def update_A(self, A: torch.Tensor):
        if A is None or A.ndim != 2:
            raise ValueError('The parameter matrix A is required and should have ndim: 2 by default')
        self.check_A_shape_validity(A=A)
        self.A = A

    def calculate_b_prime(self, b: int = None):
        b = b if b is not None else self.b
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert self.A is not None and b is not None and self.A.size(0) == b
            return self.A.size(1)
        else:
            return b

    def calculate_m_prime(self, m: int = None):
        m = m if m is not None else self.m
        if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert self.A is not None and m is not None and self.A.size(0) == m
            return self.A.size(1)
        else:
            return m

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        assert self.A is not None and self.require_data is False and self.require_parameters is False
        return self.A


class constant_c_interdependence(constant_interdependence):

    def __init__(
        self,
        b: int, m: int,
        b_prime: int = None, m_prime: int = None,
        c: float | int = 1.0,
        name: str = 'constant_c_interdependence',
        interdependence_type: str = 'attribute',
        device: str = 'cpu',
        *args, **kwargs
    ):
        if interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert b_prime is not None
            A = c * torch.ones((b, b_prime), device=device)
        elif interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert m_prime is not None
            A = c * torch.ones((m, m_prime), device=device)
        else:
            raise ValueError(f'Interdependence type {interdependence_type} is not supported')
        super().__init__(b=b, m=m, A=A, name=name, interdependence_type=interdependence_type, device=device, *args, **kwargs)


class zero_interdependence(constant_c_interdependence):

    def __init__(self, name: str = 'zero_interdependence', *args, **kwargs):
        super().__init__(c=0.0, name=name, *args, **kwargs)


class one_interdependence(constant_c_interdependence):

    def __init__(self, name: str = 'one_interdependence', *args, **kwargs):
        super().__init__(c=1.0, name=name, *args, **kwargs)


class identity_interdependence(constant_interdependence):

    def __init__(
        self,
        b: int, m: int,
        b_prime: int = None, m_prime: int = None,
        name: str = 'identity_interdependence',
        interdependence_type: str = 'attribute',
        device: str = 'cpu',
        *args, **kwargs
    ):
        if interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert b_prime is not None
            A = torch.eye(b, b_prime, device=device)
            if b != b_prime:
                warnings.warn("b and b_prime are different, this function will change the row dimensions of the inputs and cannot guarantee identity interdependence...")
        elif interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert m_prime is not None
            A = torch.eye(m, m_prime, device=device)
            if m != m_prime:
                warnings.warn("m and m_prime are different, this function will change the column dimensions of the inputs and cannot guarantee identity interdependence...")

        else:
            raise ValueError(f'Interdependence type {interdependence_type} is not supported')
        super().__init__(b=b, m=m, A=A, name=name, interdependence_type=interdependence_type, device=device, *args, **kwargs)
