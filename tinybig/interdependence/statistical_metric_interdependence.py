# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

######################################
# Statistical Metric Interdependence #
######################################

import torch
from typing import Callable

from tinybig.interdependence import interdependence
from tinybig.koala.statistics import (
    batch_kl_divergence,
    batch_pearson_correlation,
    batch_rv_coefficient,
    batch_mutual_information
)


class metric_based_interdependence(interdependence):

    def __init__(self, metric: Callable, name: str = 'metric_based_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        if metric is None:
            raise ValueError('the metric is required for the metric based interdependence function')
        self.metric = metric

    def forward(self, x: torch.Tensor, device: str = 'cpu', *args, **kwargs):
        assert x is not None and x.ndim == 2
        b, o = x.shape

        x = self.pre_process(x=x, device=device)
        A = self.metric(x)

        assert A.shape == (o, o)
        return self.post_process(x=A, device=device)


class kl_divergence_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'kl_divergence_interdependence', *args, **kwargs):
        super().__init__(metric=batch_kl_divergence, name=name, *args, **kwargs)


class pearson_correlation_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'pearson_correlation_interdependence', *args, **kwargs):
        super().__init__(metric=batch_pearson_correlation, name=name, *args, **kwargs)


class rv_coefficient_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'rv_coefficient_interdependence', *args, **kwargs):
        super().__init__(metric=batch_rv_coefficient, name=name, *args, **kwargs)


class mutual_information_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'mutual_information_interdependence', *args, **kwargs):
        super().__init__(metric=batch_mutual_information, name=name, *args, **kwargs)


