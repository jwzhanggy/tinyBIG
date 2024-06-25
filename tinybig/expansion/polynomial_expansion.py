# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import warnings
import numpy as np
import torch.nn

from tinybig.module.transformation import base_transformation as base_expansion

###################################################
# Expansions defined with closed-form polynomials #
###################################################


class taylor_expansion(base_expansion):
    def __init__(self, name='taylor_expansion', d=2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        D = sum([m**i for i in range(1, self.d+1)])
        if D > 10**7:
            warnings.warn('You have expanded the input data to a very high-dimensional representation, '
                          'with more than 10M features per instance...', UserWarning)
        return D

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x_powers = torch.ones(size=[x.size(0), 1]).to(device)
        expansion = torch.Tensor([]).to(device)

        for i in range(1, self.d+1):
            x_powers = torch.einsum('ba,bc->bac', x_powers.clone(), x).view(x_powers.size(0), x_powers.size(1)*x.size(1))
            expansion = torch.cat((expansion, x_powers), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class fourier_expansion(base_expansion):
    def __init__(self, name='fourier_expansion', P=1, N=5, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.P = P
        self.N = N

    def calculate_D(self, m: int):
        return m * self.N * 2

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        expansion = torch.Tensor([]).to(device)
        for n in range(1, self.N+1):
            cos = torch.cos(2 * np.pi * (n / self.P) * x)
            sin = torch.sin(2 * np.pi * (n / self.P) * x)
            expansion = torch.cat((expansion, cos, sin), dim=1)
        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


