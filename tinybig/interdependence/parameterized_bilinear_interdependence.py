# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################################
# Parameterized Bilinear Interdependence #
##########################################

import torch

from tinybig.interdependence import interdependence

from tinybig.reconciliation import (
    lorr_reconciliation,
    hm_reconciliation,
    lphm_reconciliation,
    dual_lphm_reconciliation,
    random_matrix_adaption_reconciliation
)


class parameterized_bilinear_interdependence(interdependence):
    def __init__(
        self,
        b: int, m: int,
        interdependence_type: str = 'attribute',
        name: str = 'parameterized_bilinear_interdependence',
        require_parameters: bool = True,
        require_data: bool = True,
        device: str = 'cpu', *args, **kwargs
    ):
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data, require_parameters=require_parameters, device=device, *args, **kwargs)
        self.parameter_fabrication = None

    def calculate_l(self):
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            if self.parameter_fabrication is None:
                return self.m ** 2
            else:
                return self.parameter_fabrication.calculate_l(n=self.m, D=self.m)
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            if self.parameter_fabrication is None:
                return self.b ** 2
            else:
                return self.parameter_fabrication.calculate_l(n=self.b, D=self.b)
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            assert x is not None and x.ndim == 2
            assert w is not None and w.ndim == 2 and w.numel() == self.calculate_l()

            x = self.pre_process(x=x, device=device)

            if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                # for instance interdependence, the parameter for calculating x.t*W*x will have dimension m*m'
                d, d_prime = self.m, self.calculate_m_prime()
            elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                # for attribute interdependence, the parameter for calculating x.t*W*x will have dimension b*b'
                d, d_prime = self.b, self.calculate_b_prime()
            else:
                raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

            if self.parameter_fabrication is None:
                W = w.reshape(d, d_prime).to(device=device)
            else:
                W = self.parameter_fabrication(w=w, n=d, D=d_prime, device=device)

            A = torch.matmul(x.t(), torch.matmul(W, x))

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A
            return A


class lowrank_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    def __init__(self, r: int = 2, name: str = 'lowrank_parameterized_bilinear_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = lorr_reconciliation(r=self.r)


class hm_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    def __init__(self, p: int, q: int = None, name: str = 'hm_parameterized_bilinear_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = hm_reconciliation(p=self.p, q=self.q)


class lphm_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    def __init__(self, r: int, p: int, q: int = None, name: str = 'lphm_parameterized_bilinear_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class dual_lphm_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    def __init__(self, r: int, p: int, q: int = None, name: str = 'dual_lphm_parameterized_bilinear_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = dual_lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class random_matrix_adaption_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    def __init__(self, r: int = 2, name: str = 'random_matrix_adaption_parameterized_bilinear_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = random_matrix_adaption_reconciliation(r=self.r)
