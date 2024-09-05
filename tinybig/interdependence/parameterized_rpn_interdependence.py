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
        data_transformation: base_transformation,
        parameter_fabrication: base_fabrication,
        remainder: base_remainder,
        name: str = 'parameterized_rpn_interdependence',
        require_parameters: bool = True, *args, **kwargs
    ):
        super().__init__(name=name, require_parameters=require_parameters, *args, **kwargs)
        if data_transformation is None or parameter_fabrication is None:
            raise ValueError('data_transformation or parameter_fabrication must be provided')
        self.data_transformation = data_transformation
        self.parameter_fabrication = parameter_fabrication
        self.remainder = remainder

    def calculate_l(self, b: int):
        assert self.o is not None and self.o_prime is not None
        D = self.data_transformation.calculate_D(m=b*self.o)
        return self.parameter_fabrication.calculate_l(n=self.o*self.o_prime, D=D)

    def forward(self, x: torch.Tensor, w: torch.nn.Parameter, b: torch.nn.Parameter = None, w_prime: torch.nn.Parameter = None, b_prime: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        assert self.o is not None and self.o_prime is not None
        x = self.pre_process(x=x, device=device)
        b, o = x.shape()
        assert o == self.o

        self.data_transformation.to(device)
        self.parameter_fabrication.to(device)
        self.remainder.to(device)

        assert self.data_transformation is not None
        x = x.reshape(1, -1)
        kappa_x = self.data_transformation(x, device=device)
        D = self.data_transformation.calculate_D(m=b*o)

        assert self.parameter_fabrication is not None
        phi_w = self.parameter_fabrication(w=w, n=self.o*self.o_prime, D=D, device=device)
        result = F.linear(kappa_x, phi_w, bias=b)

        if self.remainder is not None:
            pi_x = self.remainder(x=x, w=self.w_prime, b=b_prime, m=b*o, n=self.o*self.o_prime, device=device).to(result.device)
            result += pi_x
        A = result.reshape(self.o, -1)

        assert A.shape == (self.o, self.o_prime)
        return self.post_process(x=A, device=device)