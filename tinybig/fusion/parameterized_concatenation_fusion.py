# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

######################################################
# Parameterized Concatenation based Fusion Functions #
######################################################

import torch

from tinybig.fusion import fusion
from tinybig.reconciliation import (
    lorr_reconciliation,
    hm_reconciliation,
    lphm_reconciliation,
    dual_lphm_reconciliation,
    random_matrix_adaption_reconciliation
)


class parameterized_concatenation_fusion(fusion):

    def __init__(self, n: int = None, dims: list[int] | tuple[int] = None, name: str = "parameterized_concatenation_fusion", require_parameters: bool = True, *args, **kwargs):
        super().__init__(dims=dims, name=name, require_parameters=True, *args, **kwargs)
        if n is not None:
            self.n = n
        else:
            assert dims is not None and all([dim == dims[0] for dim in dims])
            self.n = dims[0]
        self.parameter_fabrication = None

    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        if self.n is not None:
            return self.n
        else:
            dims = dims if dims is not None else self.dims
            assert dims is not None and all([dim == dims[0] for dim in dims])
            return dims[0]

    def calculate_l(self, *args, **kwargs):
        if self.dims is None or self.n is None:
            raise ValueError("The output dimension n is required...")
        if self.parameter_fabrication is None:
            return sum(self.dims) * self.n
        else:
            return self.parameter_fabrication.calculate_l(n=self.n, D=sum(self.dims))

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor], w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not x:
            raise ValueError("The input x cannot be empty...")
        if not all(x[0].shape[:-1] == t.shape[:-1] for t in x):
            raise ValueError("Excluding the last dimension, the input x contains elements of different shapes for other dimensions...")

        if all(x[0].shape == t.shape for t in x):
            # if they are all the same shape, it will allow some cross-channel pre-processing operators...
            x = torch.stack(x, dim=0)
            x = self.pre_process(x=x, device=device)
            x = [t.squeeze(dim=0) for t in x.split(1, dim=0)]
        else:
            # otherwise, we cannot perform cross channel preprocessing, and have to pre-process them individually...
            x = [self.pre_process(t, device=device) for t in x]

        x = torch.cat(x, dim=-1)

        if self.dims is None or self.n is None:
            raise ValueError("The output dimension n is required...")
        if self.parameter_fabrication is None:
            W = w.reshape(self.n, sum(self.dims)).to(device=device)
        else:
            W = self.parameter_fabrication(w=w, n=self.n, D=sum(self.dims), device=device)

        fused_x = torch.matmul(x, W.t())

        assert fused_x.size(-1) == self.calculate_n([element.size(-1) for element in x])
        return self.post_process(x=fused_x, device=device)


class lowrank_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    def __init__(self, r: int = 2, name: str = 'lowrank_parameterized_concatenation_fusion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = lorr_reconciliation(r=self.r)


class hm_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    def __init__(self, p: int, q: int = None, name: str = 'hm_parameterized_concatenation_fusion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p
        assert self.n is not None and self.n % self.p == 0
        assert self.dims is not None and sum(self.dims) % self.q == 0
        self.parameter_fabrication = hm_reconciliation(p=self.p, q=self.q)


class lphm_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    def __init__(self, r: int, p: int, q: int = None, name: str = 'lphm_parameterized_concatenation_fusion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.p = p
        self.q = q if q is not None else p
        assert self.n is not None and self.n % self.p == 0
        assert self.dims is not None and sum(self.dims) % self.q == 0
        self.parameter_fabrication = lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class dual_lphm_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    def __init__(self, r: int, p: int, q: int = None, name: str = 'dual_lphm_parameterized_concatenation_fusion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.p = p
        self.q = q if q is not None else p
        assert self.n is not None and self.n % self.p == 0
        assert self.dims is not None and sum(self.dims) % self.q == 0
        self.parameter_fabrication = dual_lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class random_matrix_adaption_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    def __init__(self, r: int = 2, name: str = 'random_matrix_adaption_parameterized_concatenation_fusion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = random_matrix_adaption_reconciliation(r=self.r)
