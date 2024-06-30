# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch.nn

from tinybig.expansion import expansion


#####################################################
# Expansions defined with recursively defined basis #
#####################################################

class bspline_expansion(expansion):
    def __init__(self, name='bspline_expansion', grid_range=(-1, 1), t=5, d=3, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.grid_range = grid_range
        self.t = t
        self.d = d
        self.grid = None
        self.m = None

    def calculate_D(self, m: int):
        return m * (self.t + self.d)

    def initialize_grid(self, m: int, device='cpu', grid_range=None, t=None, d=None):
        self.m = m
        grid_range = grid_range if grid_range is not None else self.grid_range
        t = t if t is not None else self.t
        d = d if d is not None else self.d

        h = (grid_range[1] - grid_range[0]) / t
        self.grid = torch.Tensor((torch.arange(-d, t + d + 1) * h + grid_range[0])
                                 .expand(m, -1).contiguous()).to(device)

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        if self.grid is None:
            self.initialize_grid(m=x.size(1), device=device, *args, **kwargs)
        assert x.dim() == 2 and x.size(1) == self.m
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.d + 1):
            bases = (((x - self.grid[:, : -(k + 1)]) / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)]) * bases[:, :, :-1]) +
                     ((self.grid[:, k + 1:] - x) / (self.grid[:, k + 1:] - self.grid[:, 1:(-k)]) * bases[:, :, 1:]))
        expansion = bases.contiguous().view(x.size(0), -1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class chebyshev_expansion(expansion):
    def __init__(self, name='chebyshev_polynomial_expansion', d=5, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        return m * self.d

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d+1]).to(device)
        if self.d > 0:
            expansion[:,:,1] = x
        for n in range(2, self.d+1):
            expansion[:, :, n] = 2 * x * expansion[:, :, n-1].clone() - expansion[:, :, n-2].clone()
        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)
        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class jacobi_expansion(expansion):
    def __init__(self, name='jacobi_polynomial_expansion', d=5, alpha=1.0, beta=1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d
        self.alpha = alpha
        self.beta = beta

    def calculate_D(self, m: int):
        return m * self.d

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d+1]).to(device)
        if self.d > 0:
            expansion[:,:,1] = ((self.alpha-self.beta) + (self.alpha+self.beta+2) * x) / 2
        for n in range(2, self.d+1):
            coeff_1 = 2*n*(n+self.alpha+self.beta)*(2*n+self.alpha+self.beta-2)
            coeff_2 = (2*n+self.alpha+self.beta-1)*(2*n+self.alpha+self.beta)*(2*n+self.alpha+self.beta-2)
            coeff_3 = (2*n+self.alpha+self.beta-1)*(self.alpha**2-self.beta**2)
            coeff_4 = 2*(n+self.alpha-1)*(n+self.beta-1)*(2*n+self.alpha+self.beta)
            expansion[:,:,n] = ((coeff_2/coeff_1)*x + coeff_3/coeff_1)*expansion[:,:,n-1].clone() - (coeff_4/coeff_1)*expansion[:,:,n-2].clone()
        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)

