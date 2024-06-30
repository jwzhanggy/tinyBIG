# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch
import torch.nn.functional as F

from tinybig.remainder import remainder
from tinybig.util.util import func_x


####################
# Basic Remainders #
####################


class constant_remainder(remainder):
    def __init__(self, name='constant_remainder', c=1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.c = c

    def __call__(self, n: int, device='cpu', *args, **kwargs):
        x = self.c * torch.ones(n)
        return self.activation(x=x, device=device)


class zero_remainder(constant_remainder):
    def __init__(self, name='zero_remainder', *args, **kwargs):
        super().__init__(name=name, c=0.0, *args, **kwargs)


class one_remainder(constant_remainder):
    def __init__(self, name='one_remainder', *args, **kwargs):
        super().__init__(name=name, c=1.0, *args, **kwargs)


class identity_remainder(remainder):
    def __init__(self, name='identity_remainder', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return self.activation(x=x, device=device)


class linear_remainder(remainder):
    def __init__(self, name='linear_remainder', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def __call__(self, x: torch.Tensor, w=None, b=None, device='cpu', *args, **kwargs):
        if w is not None:
            x = F.linear(x, w, bias=b)
        return self.activation(x=x, device=device)