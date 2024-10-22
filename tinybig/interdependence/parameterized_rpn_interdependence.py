# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################
# Parameterized RPN Interdependence #
#####################################

import torch
import torch.nn.functional as F

from tinybig.interdependence import interdependence
import tinybig.module.base_transformation as base_transformation
import tinybig.module.base_fabrication as base_fabrication
import tinybig.module.base_remainder as base_remainder


class parameterized_rpn_interdependence(interdependence):

    def __init__(
        self,
        b: int, m: int,
        data_transformation: base_transformation,
        parameter_fabrication: base_fabrication,
        b_prime: int = None, m_prime: int = None,
        interdependence_type: str = 'attribute',
        name: str = 'parameterized_rpn_interdependence',
        require_parameters: bool = True,
        require_data: bool = True,
        device: str = 'cpu', *args, **kwargs
    ):
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data,
                         require_parameters=require_parameters, device=device, *args, **kwargs)

        if data_transformation is None or parameter_fabrication is None:
            raise ValueError('data_transformation or parameter_fabrication must be specified...')

        self.data_transformation = data_transformation
        self.parameter_fabrication = parameter_fabrication
        self.b_prime = b_prime if b_prime is not None else b
        self.m_prime = m_prime if m_prime is not None else m

    def calculate_l(self):
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.m, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.b, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

        assert d is not None and d_prime is not None
        D = self.data_transformation.calculate_D(m=d)
        return self.parameter_fabrication.calculate_l(n=d_prime, D=D)

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
            assert x is not None and x.ndim == 2
            assert w is not None and w.ndim == 2 and w.numel() == self.calculate_l()

            x = self.pre_process(x=x, device=device)

            self.data_transformation.to(device)
            self.parameter_fabrication.to(device)

            if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                d, d_prime = self.m, self.calculate_b_prime()
            elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                d, d_prime = self.b, self.calculate_m_prime()
            else:
                raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

            kappa_x = self.data_transformation(x.t(), device=device)
            D = self.data_transformation.calculate_D(m=d)
            phi_w = self.parameter_fabrication(w=w, n=d_prime, D=D, device=device)

            A = torch.matmul(kappa_x, phi_w.T)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A