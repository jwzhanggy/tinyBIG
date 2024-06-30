# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinybig.reconciliation import reconciliation


########################
# Hypernet reconciliation #
########################

class hypernet_reconciliation(reconciliation):
    def __init__(self, name='hypernet_reconciliation', l: int = 64, hidden_dim: int = 128, static: bool = True, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.l = l
        self.hidden_dim = hidden_dim
        warnings.warn('In hypernet based reconciliation function, parameter l and hidden_dim cannot be None, '
                      'which will be set with the default values 64 and 128, respectively...')
        self.net = None
        self.static = static

    def calculate_l(self, n: int, D: int):
        return self.l

    def initialize_hypernet(self, l: int, n: int, D: int, hidden_dim: int, static=True, device='cpu'):
        self.net = nn.Sequential(
            nn.Linear(l, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, n*D)
        ).to(device)

        for param in self.net.parameters():
            torch.nn.init.normal_(param)

        if static:
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.net.parameters():
                param.detach()
        else:
            for param in self.net.parameters():
                param.requires_grad = True

    def __call__2(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        print(w.shape, self.calculate_l(n, D))
        return F.linear(w, torch.ones(n*D, self.calculate_l(n, D)).to(device)).view(n, D).to(device)

    def __call__(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        assert w.dim() == 2 and w.size(1) == self.calculate_l(n=n, D=D)
        if self.net is None:
            self.initialize_hypernet(l=self.calculate_l(n, D), n=n, D=D, hidden_dim=self.hidden_dim, static=self.static, device=device)
        return self.net(w).view(n, D)
