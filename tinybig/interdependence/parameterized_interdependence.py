# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################
# Parameterized Interdependence #
#################################

import torch

from tinybig.interdependence import interdependence
from tinybig.reconciliation import (
    lorr_reconciliation,
    hm_reconciliation,
    lphm_reconciliation,
    dual_lphm_reconciliation,
    random_matrix_adaption_reconciliation
)


class parameterized_interdependence(interdependence):
    def __init__(
        self,
        o: int,
        o_prime: int = None,
        name: str = 'parameterized_interdependence',
        require_parameters: bool = True,
        *args, **kwargs
    ):
        super().__init__(name=name, require_parameters=require_parameters, *args, **kwargs)
        self.o = o
        self.o_prime = o_prime if o_prime is not None else o
        self.parameter_fabrication = None

    def calculate_l(self):
        assert self.o is not None and self.o_prime is not None
        if self.parameter_fabrication is None:
            return self.o * self.o_prime
        else:
            return self.parameter_fabrication.calculate_l(n=self.o, D=self.o_prime)

    def forward(self, w: torch.nn.Parameter, device: str = 'cpu', *args, **kwargs):
        assert self.o is not None and self.o_prime is not None

        assert w.ndim == 2 and w.numel() == self.calculate_l()
        if self.parameter_fabrication is None:
            A = w.reshape(self.o, self.o_prime).to(device=device)
        else:
            A = self.parameter_fabrication(w=w, n=self.o, D=self.o_prime, device=device)

        assert A.shape == (self.o, self.o_prime)
        return self.post_process(x=A, device=device)


class lowrank_parameterized_interdependence(parameterized_interdependence):
    def __init__(self, r: int = 2, name: str = 'lowrank_parameterized_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = lorr_reconciliation(r=self.r)


class hm_parameterized_interdependence(parameterized_interdependence):
    def __init__(self, p: int, q: int = None, name: str = 'hm_parameterized_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.p = p
        self.q = q if q is not None else p
        assert self.o % self.p == 0 and self.o_prime % self.q == 0

        self.parameter_fabrication = hm_reconciliation(p=self.p, q=self.q)


class lphm_parameterized_interdependence(parameterized_interdependence):
    def __init__(self, r: int, p: int, q: int = None, name: str = 'lphm_parameterized_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p
        assert self.o % self.p == 0 and self.o_prime % self.q == 0

        self.parameter_fabrication = lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class dual_lphm_parameterized_interdependence(parameterized_interdependence):
    def __init__(self, r: int, p: int, q: int = None, name: str = 'dual_lphm_parameterized_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p
        assert self.o % self.p == 0 and self.o_prime % self.q == 0

        self.parameter_fabrication = dual_lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class random_matrix_adaption_parameterized_interdependence(parameterized_interdependence):
    def __init__(self, r: int = 2, name: str = 'random_matrix_adaption_parameterized_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = random_matrix_adaption_reconciliation(r=self.r)
