# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################
# Metric based Compression Function #
#####################################

import torch

from typing import Callable

from tinybig.compression import transformation
from tinybig.koala.statistics import batch_mean, batch_weighted_mean, batch_std, batch_mode, batch_median, batch_entropy, batch_variance, batch_skewness, batch_harmonic_mean, batch_geometric_mean
from tinybig.koala.linear_algebra import batch_sum, batch_norm, batch_prod, batch_min, batch_max, batch_l1_norm, batch_l2_norm


class metric_compression(transformation):

    def __init__(self, name: str = 'metric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.metric = metric

    def calculate_D(self, m: int):
        return 1

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        assert self.metric is not None
        compression = self.metric(x).unsqueeze(1)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class max_compression(metric_compression):
    def __init__(self, name: str = 'max_compression', *args, **kwargs):
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class min_compression(metric_compression):
    def __init__(self, name: str = 'min_compression', *args, **kwargs):
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class sum_compression(metric_compression):
    def __init__(self, name: str = 'sum_compression', *args, **kwargs):
        super().__init__(name=name, metric=batch_sum, *args, **kwargs)


class mean_compression(metric_compression):
    def __init__(self, name: str = 'mean_compression', *args, **kwargs):
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)


class prod_compression(metric_compression):
    def __init__(self, name: str = 'prod_compression', *args, **kwargs):
        super().__init__(name=name, metric=batch_prod, *args, **kwargs)


class median_compression(metric_compression):
    def __init__(self, name: str = 'median_compression', *args, **kwargs):
        super().__init__(name=name, metric=batch_median, *args, **kwargs)
