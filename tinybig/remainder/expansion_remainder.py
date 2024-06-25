# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch
import torch.nn.functional as F

from tinybig.module.remainder import base_remainder


##############################
# Expansion based Remainders #
##############################


class bspline_remainder(base_remainder):
    def __init__(self, name='bspline_remainder', grid_range=(-1, 1), num_interval=5, spline_order=3, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.grid_range = grid_range
        self.num_interval = num_interval
        self.spline_order = spline_order
        self.grid = None
        self.m = None
        self.reconciling_parameters = None

    def calculate_D(self, m: int):
        return m * (self.num_interval + self.spline_order)

    def initialize_reconciling_parameters(self, m, D, r=2, device='cpu'):
        self.reconciling_parameters = torch.nn.Parameter(torch.rand(m, D)).to(device)
        torch.nn.init.xavier_uniform_(self.reconciling_parameters)

    def initialize_grid(self, m: int, grid_range=None, num_interval=None, spline_order=None, device='cpu', *args, **kwargs):
        self.m = m
        grid_range = grid_range if grid_range is not None else self.grid_range
        num_interval = num_interval if num_interval is not None else self.num_interval
        spline_order = spline_order if spline_order is not None else self.spline_order

        h = (grid_range[1] - grid_range[0]) / num_interval
        self.grid = torch.Tensor((torch.arange(-spline_order, num_interval + spline_order + 1) * h + grid_range[0])
                                 .expand(m, -1).contiguous()).to(device)

    def __call__(self, x: torch.Tensor, w, b, device='cpu', *args, **kwargs):
        if self.grid is None:
            self.initialize_grid(device=device, *args, **kwargs)
        if self.require_parameters and self.reconciling_parameters is None:
            self.initialize_reconciling_parameters(m=x.size(1), D=self.calculate_D(x.size(1)), device=device)
        assert x.dim() == 2 and x.size(1) == self.m
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (((x - self.grid[:, : -(k + 1)]) / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)]) * bases[:, :, :-1]) +
                     ((self.grid[:, k + 1:] - x) / (self.grid[:, k + 1:] - self.grid[:, 1:(-k)]) * bases[:, :, 1:]))
        assert bases.size() == (x.size(0), self.m, self.num_interval + self.spline_order)
        return F.linear(F.linear(bases.contiguous().view(x.size(0), -1), self.reconciling_parameters), w, bias=b)