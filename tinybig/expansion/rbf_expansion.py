# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch.nn

from tinybig.expansion import expansion


#########################################################
# Expansions defined with RBF for given base fix points #
#########################################################

class gaussian_rbf_expansion(expansion):
    def __init__(self, name='gaussian_rbf_expansion', base_range=(-1, 1), num_interval=10, epsilon=1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.betaase_range = base_range
        self.num_interval = num_interval
        self.epsilon = epsilon
        self.base = None
        self.m = None

    def calculate_D(self, m: int):
        return m * self.num_interval

    def initialize_base(self, m: int, device='cpu', base_range=None, num_interval=None):
        self.m = m
        base_range = base_range if base_range is not None else self.betaase_range
        num_interval = num_interval if num_interval is not None else self.num_interval
        self.base = torch.Tensor(torch.linspace(base_range[0], base_range[1], num_interval)).to(device)

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        if self.base is None:
            self.initialize_base(m=x.size(1), device=device, *args, **kwargs)
        assert x.dim() == 2 and x.size(1) == self.m
        expansion = torch.exp(-((x[..., None] - self.base) * self.epsilon) ** 2).view(x.size(0), -1)
        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class inverse_quadratic_rbf_expansion(gaussian_rbf_expansion):

    def __init__(self, name='inverse_quadratic_rbf', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        if self.base is None:
            self.initialize_base(m=x.size(1), device=device, *args, **kwargs)
        assert x.dim() == 2 and x.size(1) == self.m
        expansion = (1/(1+((x[..., None] - self.base) * self.epsilon) ** 2)).view(x.size(0), -1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)

