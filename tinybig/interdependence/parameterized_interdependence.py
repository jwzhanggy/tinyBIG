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
        b: int, m: int,
        b_prime: int = None, m_prime: int = None,
        interdependence_type: str = 'attribute',
        name: str = 'parameterized_interdependence',
        require_parameters: bool = True,
        require_data: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data, require_parameters=require_parameters, device=device, *args, **kwargs)
        self.parameter_fabrication = None
        self.b_prime = b_prime if b_prime is not None else b
        self.m_prime = m_prime if m_prime is not None else m

    def calculate_l(self):
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

        assert d is not None and d_prime is not None
        if self.parameter_fabrication is None:
            return d * d_prime
        else:
            return self.parameter_fabrication.calculate_l(n=d, D=d_prime)

    def calculate_b_prime(self, b: int = None):
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence'] and self.b_prime is not None:
            return self.b_prime
        else:
            return b if b is not None else self.b

    def calculate_m_prime(self, m: int = None):
        if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence'] and self.m_prime is not None:
            return self.m_prime
        else:
            return m if m is not None else self.m

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            assert w.ndim == 2 and w.numel() == self.calculate_l()

            if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                d, d_prime = self.b, self.calculate_b_prime()
            elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                d, d_prime = self.m, self.calculate_m_prime()
            else:
                raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

            if self.parameter_fabrication is None:
                A = w.reshape(d, d_prime).to(device=device)
            else:
                A = self.parameter_fabrication(w=w, n=d, D=d_prime, device=device)
            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A
            return A


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

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = hm_reconciliation(p=self.p, q=self.q)


class lphm_parameterized_interdependence(parameterized_interdependence):
    def __init__(self, r: int, p: int, q: int = None, name: str = 'lphm_parameterized_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class dual_lphm_parameterized_interdependence(parameterized_interdependence):
    def __init__(self, r: int, p: int, q: int = None, name: str = 'dual_lphm_parameterized_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = dual_lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class random_matrix_adaption_parameterized_interdependence(parameterized_interdependence):
    def __init__(self, r: int = 2, name: str = 'random_matrix_adaption_parameterized_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = random_matrix_adaption_reconciliation(r=self.r)
