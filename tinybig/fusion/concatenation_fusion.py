# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

########################################
# Concatenation based Fusion Functions #
########################################

import torch
from tinybig.fusion import fusion


class concatenation_fusion(fusion):

    def __init__(self, dims: list[int] | tuple[int] = None, name: str = "concatenation_fusion", require_parameters: bool = False, *args, **kwargs):
        super().__init__(dims=dims, name=name, require_parameters=False, *args, **kwargs)

    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        dims = dims if dims is not None else self.dims
        assert dims is not None
        return sum(dims)

    def calculate_l(self, *args, **kwargs):
        return 0

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

        fused_x = torch.cat(x, dim=-1)

        assert fused_x.size(-1) == self.calculate_n([element.size(-1) for element in x])
        return self.post_process(x=fused_x, device=device)
