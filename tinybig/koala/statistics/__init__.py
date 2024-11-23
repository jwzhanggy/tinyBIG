# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

######################
# Statistics Library #
######################

"""

This module provides the libraries of "statistics" that can be used to build the RPN model within the tinyBIG toolkit.

## Statistics Library

Statistics is discipline that serves as the lens through which we make sense of our data-driven world,
transforming raw numbers into meaningful insights and informed decisions.

## Functions Implementation

Currently, the functions implemented in this library include:

* metrics, and batch metrics
* kernels, and batch kernels

"""

from tinybig.koala.statistics.metric import (
    metric,
    mean, batch_mean,
    geometric_mean, batch_geometric_mean,
    harmonic_mean, batch_harmonic_mean,
    weighted_mean, batch_weighted_mean,
    median, batch_median,
    mode, batch_mode,
    std, batch_std,
    entropy, batch_entropy,
    variance, batch_variance,
    skewness, batch_skewness
)

from tinybig.koala.statistics.kernel import (
    kernel,

    kl_divergence_kernel,
    batch_kl_divergence_kernel,

    pearson_correlation_kernel,
    batch_pearson_correlation_kernel,

    rv_coefficient_kernel,
    batch_rv_coefficient_kernel,

    mutual_information_kernel,
    batch_mutual_information_kernel,

    custom_hybrid_kernel,
    batch_custom_hybrid_kernel,
)