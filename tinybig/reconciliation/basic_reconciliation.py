# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import warnings
import torch

from tinybig.module.fabrication import base_fabrication as base_reconciliation


#####################
# Basic reconciliation #
#####################

class constant_reconciliation(base_reconciliation):
    def __init__(self, name='constant_reconciliation', c=1.0, *args, **kwargs):
        super().__init__(name=name, require_parameters=False, *args, **kwargs)
        self.c = c

    def calculate_l(self, n: int, D: int):
        return 0

    def __call__(self, n: int, D: int, w: torch.nn.Parameter = None, device='cpu', *args, **kwargs):
        return self.c * torch.ones(n, D).to(device)


class constant_eye_reconciliation(base_reconciliation):
    def __init__(self, name='constant_c_reconciliation', *args, **kwargs):
        super().__init__(name=name, require_parameters=False, *args, **kwargs)

    def calculate_l(self, n: int, D: int):
        return 0

    def __call__(self, n: int, D: int, w: torch.nn.Parameter = None, device='cpu', *args, **kwargs):
        return torch.eye(n=n, m=D).to(device)


class identity_reconciliation(base_reconciliation):
    def __init__(self, name='identity_reconciliation', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_l(self, n: int, D: int):
        return n*D

    def __call__(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        return w.view(n, D).to(device)


class masking_reconciliation(base_reconciliation):

    def __init__(self, name='masking_reconciliation', masking_ratio=0.5, fixed_mask: bool = True, *args, **kwargs):
        # masking ratio: used parameter percentage
        # e.g., masking_ratio=1.0: all parameters are used; masking_ratio=0.0: no parameters
        super().__init__(name=name, *args, **kwargs)
        self.masking_ratio = masking_ratio
        self.mask_matrix = None
        self.fixed_mask = fixed_mask

    def calculate_l(self, n: int, D: int):
        return n * D

    def generate_masking_matrix(self, n, D):
        self.mask_matrix = torch.rand(n, D) < self.masking_ratio

    def __call__(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        if not self.fixed_mask:
            self.generate_masking_matrix(n=n, D=D)
        else:
            if self.mask_matrix is None:
                self.generate_masking_matrix(n=n, D=D)
        return w.view(n, D) * self.mask_matrix.to(device)


class duplicated_padding_reconciliation(base_reconciliation):
    def __init__(self, name='duplicated_padding_reconciliation', p=2, q=None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p

    def calculate_l(self, n: int, D: int):
        s, t = int(n/self.p), int(D/self.q)
        assert (self.p * self.q * s * t == n * D)
        return s * t

    def __call__(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        assert w.dim() == 2 and w.size(1) == self.calculate_l(n=n, D=D)
        s, t = int(n / self.p), int(D / self.q)
        A = torch.ones(self.p, self.q).view(1, -1).to(device)
        return torch.einsum('pq,st->psqt', A, w).view(self.p*s, self.q*t).to(device)