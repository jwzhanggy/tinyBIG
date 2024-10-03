# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################################
# Multi-Head & Multi-Channel Fusion Functions #
###############################################


from tinybig.module.base_fusion import (
    fusion
)

from tinybig.fusion.basic_fusion import (
    weighted_summation_fusion,
    summation_fusion,
    average_fusion,
    parameterized_weighted_summation_fusion
)

from tinybig.fusion.metric_fusion import (
    metric_fusion,
    mean_fusion,
    prod_fusion,
    max_fusion,
    min_fusion,
    median_fusion,
    sum_fusion,
)

from tinybig.fusion.concatenation_fusion import (
    concatenation_fusion,
)

from tinybig.fusion.parameterized_concatenation_fusion import (
    parameterized_concatenation_fusion,
    lowrank_parameterized_concatenation_fusion,
    hm_parameterized_concatenation_fusion,
    lphm_parameterized_concatenation_fusion,
    dual_lphm_parameterized_concatenation_fusion,
    random_matrix_adaption_parameterized_concatenation_fusion
)

