# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################################
# Multi-Head & Multi-Channel Fusion Functions #
###############################################

r"""

This module provides the "fusion functions" that can be used to build the RPN model within the tinyBIG toolkit.

## Fusion Function

In the tinyBIG library, we introduce several advanced fusion strategies that can more effectively aggregate the outputs from the wide architectures.
Formally, given the input matrices $\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k$, their fusion output can be represented as

$$
    \begin{equation}
    \mathbf{A} = \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k).
    \end{equation}
$$

The dimensions of the input matrices $\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k$ may be identical or vary,
depending on the specific definition of the fusion function.

## Classes in this Module

This module contains the following categories of compression functions:

* Basic fusion functions (such as summation_fusion, weighted_summation_fusion, average_fusion)
* Metric based fusion functions (such as mean_fusion, max_fusion, min_fusion, etc.)
* Concatenation based fusion function
* Parameterized concatenation based fusion functions (such as lowrank_parameterized_concatenation_fusion, etc.)
"""

from tinybig.module.base_fusion import (
    fusion
)

from tinybig.fusion.basic_fusion import (
    weighted_summation_fusion,
    summation_fusion,
    average_fusion,
    average_fusion as mean_fusion,
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

