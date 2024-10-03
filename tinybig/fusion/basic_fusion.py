# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Basic Fusion Functions #
##########################

import torch

from tinybig.fusion import fusion


class weighted_summation_fusion(fusion):

    def __init__(self, dims: list[int] | tuple[int] = None, weights: torch.Tensor = None, name: str = "weighted_summation_fusion", *args, **kwargs):
        super().__init__(dims=dims, name=name, *args, **kwargs)
        self.weights = weights

    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        dims = dims if dims is not None else self.dims
        assert dims is not None and all(dim == dims[0] for dim in dims)
        return dims[0]

    def calculate_l(self, *args, **kwargs):
        if self.require_parameters:
            return self.get_num()
        else:
            return 0

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor], w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not x:
            raise ValueError("The input x cannot be empty...")
        if not all(x[0].shape == t.shape for t in x):
            raise ValueError("The input x must have the same shape.")

        x = torch.stack(x)
        x = self.pre_process(x=x, device=device)

        if self.require_parameters:
            weights = w
        else:
            weights = self.weights

        assert x.ndim >= 1 and all(item.shape == x[0].shape for item in x)
        assert weights is not None and x.size(0) == weights.size(0)

        weights = weights.view(-1, *[1]*(len(x[0].shape)))
        fused_x = (x * weights).sum(dim=0)

        assert fused_x.size(-1) == self.calculate_n([element.size(-1) for element in x])
        return self.post_process(x=fused_x, device=device)


class summation_fusion(weighted_summation_fusion):

    def __init__(self, dims: list[int] | tuple[int], name: str = "summation_fusion", require_parameters: bool = False,  *args, **kwargs):
        super().__init__(dims=dims, weights=torch.ones(len(dims)), name=name, require_parameters=False, *args, **kwargs)


class average_fusion(weighted_summation_fusion):
    def __init__(self, dims: list[int] | tuple[int], name: str = "average_fusion", require_parameters: bool = False, *args, **kwargs):
        super().__init__(dims=dims, weights=1.0/len(dims)*torch.ones(len(dims)), name=name, require_parameters=False, *args, **kwargs)


class parameterized_weighted_summation_fusion(weighted_summation_fusion):
    def __init__(self, dims: list[int] | tuple[int] = None, name: str = "parameterized_weighted_summation_fusion", require_parameters: bool = True, *args, **kwargs):
        super().__init__(dims=dims, weights=None, name=name, require_parameters=True, *args, **kwargs)







