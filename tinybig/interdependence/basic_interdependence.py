# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#########################
# Basic Interdependence #
#########################

import torch

from tinybig.interdependence import interdependence


class constant_interdependence(interdependence):

    def __init__(self, A: torch.Tensor, name: str = 'constant_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.A = A
        if self.A is None or self.A.ndim != 2:
            print('The parameter matrix A is required and should have ndim: 2 by default')
        assert self.A is not None and self.A.ndim == 2

        self.o = self.A.size(0)
        self.o_prime = self.A.size(1)

    def update_A(self, A: torch.Tensor):
        self.A = A
        if self.A is None or self.A.ndim != 2:
            print('The parameter matrix A is required and should have ndim: 2 by default')
        assert self.A is not None and self.A.ndim == 2

        self.o = self.A.size(0)
        self.o_prime = self.A.size(1)

    def forward(self, x: torch.Tensor = None, device: str = 'cpu', *args, **kwargs):
        assert self.A.shape == (self.o, self.o_prime)
        return self.post_process(x=self.A, device=device)


class constant_c_interdependence(constant_interdependence):

    def __init__(self, c: float | int, o: int, o_prime: int, name: str = 'constant_c_interdependence', *args, **kwargs):
        self.c = c
        A = self.c * torch.ones((o, o_prime))
        super().__init__(A=A, name=name, *args, **kwargs)


class zero_interdependence(constant_c_interdependence):

    def __init__(self, o: int, o_prime: int, name: str = 'zero_interdependence', *args, **kwargs):
        super().__init__(c=0.0, o=o, o_prime=o_prime, name=name, *args, **kwargs)


class one_interdependence(constant_c_interdependence):

    def __init__(self, o: int, o_prime: int, name: str = 'one_interdependence', *args, **kwargs):
        super().__init__(c=1.0, o=o, o_prime=o_prime, name=name, *args, **kwargs)


class identity_interdependence(constant_interdependence):

    def __init__(self, o: int, o_prime: int, name: str = 'identity_interdependence', *args, **kwargs):
        A = torch.eye(o, o_prime)
        super().__init__(A=A, name=name, *args, **kwargs)
