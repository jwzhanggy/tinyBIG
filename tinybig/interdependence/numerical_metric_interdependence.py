# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################################
# Numerical Metric Interdependence #
####################################

import functools
from typing import Union, Any

from tinybig.interdependence import metric_based_interdependence
from tinybig.koala.linear_algebra import (
    batch_inner_product,
    batch_cosine_similarity,
    batch_minkowski_distance,
    batch_manhattan_distance,
    batch_euclidean_distance,
    batch_chebyshev_distance,
    batch_canberra_distance
)


class inner_product_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'inner_product_interdependence', *args, **kwargs):
        super().__init__(metric=batch_inner_product, name=name, *args, **kwargs)


class cosine_similarity_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'cosine_similarity_interdependence', *args, **kwargs):
        super().__init__(metric=batch_cosine_similarity, name=name, *args, **kwargs)


class minkowski_distance_interdependence(metric_based_interdependence):
    def __init__(self, p: Union[int, float, str, Any], name: str = 'minkowski_distance_interdependence', *args, **kwargs):
        batch_minkowski_distance_on_p = functools.partial(batch_minkowski_distance, p=p)
        super().__init__(metric=batch_minkowski_distance_on_p, name=name, *args, **kwargs)


class manhattan_distance_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'manhattan_distance_interdependence', *args, **kwargs):
        super().__init__(metric=batch_manhattan_distance, name=name, *args, **kwargs)


class euclidean_distance_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'euclidean_distance_interdependence', *args, **kwargs):
        super().__init__(metric=batch_euclidean_distance, name=name, *args, **kwargs)


class chebyshev_distance_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'chebyshev_distance_interdependence', *args, **kwargs):
        super().__init__(metric=batch_chebyshev_distance, name=name, *args, **kwargs)


class canberra_distance_interdependence(metric_based_interdependence):
    def __init__(self, name: str = 'canberra_distance_interdependence', *args, **kwargs):
        super().__init__(metric=batch_canberra_distance, name=name, *args, **kwargs)

