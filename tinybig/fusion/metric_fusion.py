# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############################################
# Numerical Operator based Fusion Functions #
#############################################

from typing import Callable
import torch

from tinybig.fusion import fusion as base_fusion
from tinybig.koala.statistics import batch_mean, batch_weighted_mean, batch_std, batch_mode, batch_median, batch_entropy, batch_variance, batch_skewness, batch_harmonic_mean, batch_geometric_mean
from tinybig.koala.linear_algebra import batch_sum, batch_norm, batch_prod, batch_min, batch_max, batch_l1_norm, batch_l2_norm


class metric_fusion(base_fusion):

    def __init__(
        self,
        dims: list[int] | tuple[int],
        metric: Callable[[torch.Tensor], torch.Tensor],
        name: str = "metric_fusion",
        *args, **kwargs
    ):
        super().__init__(dims=dims, name=name, require_parameters=False, *args, **kwargs)
        self.metric = metric

    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        dims = dims if dims is not None else self.dims
        assert dims is not None and all(dim == dims[0] for dim in dims)
        return dims[0]

    def calculate_l(self, *args, **kwargs):
        return 0

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor], w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not x:
            raise ValueError("The input x cannot be empty...")
        if not all(x[0].shape == t.shape for t in x):
            raise ValueError("The input x must have the same shape.")

        x = torch.stack(x, dim=0)
        x = self.pre_process(x=x, device=device)

        assert self.metric is not None and isinstance(self.metric, Callable)

        x_shape = x.shape
        x_permuted = x.permute(*range(1, x.ndim), 0)

        fused_x = self.metric(x_permuted.view(-1, x_shape[0])).view(x_shape[1:])

        assert fused_x.size(-1) == self.calculate_n([element.size(-1) for element in x])
        return self.post_process(x=fused_x, device=device)


class max_fusion(metric_fusion):
    def __init__(self, name: str = "max_fusion", *args, **kwargs):
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class min_fusion(metric_fusion):
    def __init__(self, name: str = "min_fusion", *args, **kwargs):
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class sum_fusion(metric_fusion):
    def __init__(self, name: str = "sum_fusion", *args, **kwargs):
        super().__init__(name=name, metric=batch_sum, *args, **kwargs)


class mean_fusion(metric_fusion):
    def __init__(self, name: str = "mean_fusion", *args, **kwargs):
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)


class prod_fusion(metric_fusion):
    def __init__(self, name: str = "prod_fusion", *args, **kwargs):
        super().__init__(name=name, metric=batch_prod, *args, **kwargs)


class median_fusion(metric_fusion):
    def __init__(self, name: str = "median_fusion", *args, **kwargs):
        super().__init__(name=name, metric=batch_median, *args, **kwargs)


class l1_norm_fusion(metric_fusion):
    def __init__(self, name: str = "l1_norm_fusion", *args, **kwargs):
        super().__init__(name=name, metric=batch_l1_norm, *args, **kwargs)


class l2_norm_fusion(metric_fusion):
    def __init__(self, name: str = "l2_norm_fusion", *args, **kwargs):
        super().__init__(name=name, metric=batch_l2_norm, *args, **kwargs)


