# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Linear Algebra Library #
##########################

"""
This module provides the libraries of "linear algebra" that can be used to build the RPN model within the tinyBIG toolkit.

## Linear Algebra Library

Linear algebra is a captivating branch of mathematics that serves as the backbone of modern scientific computation and data analysis.
At its heart, this field explores the elegant interplay between linear equations, vectors, and matrices, providing powerful tools to solve complex problems in multiple dimensions.

## Functions Implementation

Currently, the functions implemented in this library include

* metrics, and batch metrics
* kernels, and batch kernels
* matrix operations

"""

from tinybig.koala.linear_algebra.metric import (
    metric,

    norm, batch_norm,

    l1_norm,
    batch_l1_norm,

    l2_norm,
    batch_l2_norm,

    sum,
    batch_sum,

    prod,
    batch_prod,

    max,
    batch_max,

    min,
    batch_min,
)

from tinybig.koala.linear_algebra.kernel import (
    kernel,

    linear_kernel,
    instance_linear_kernel,
    batch_linear_kernel,

    polynomial_kernel,
    instance_polynomial_kernel,
    batch_polynomial_kernel,

    hyperbolic_tangent_kernel,
    instance_hyperbolic_tangent_kernel,
    batch_hyperbolic_tangent_kernel,

    cosine_similarity_kernel,
    instance_cosine_similarity_kernel,
    batch_cosine_similarity_kernel,

    minkowski_distance_kernel,
    minkowski_distance,
    instance_minkowski_distance,
    batch_minkowski_distance,

    manhattan_distance_kernel,
    manhattan_distance,
    instance_manhattan_distance,
    batch_manhattan_distance,

    euclidean_distance_kernel,
    euclidean_distance,
    instance_euclidean_distance,
    batch_euclidean_distance,

    chebyshev_distance_kernel,
    chebyshev_distance,
    instance_chebyshev_distance,
    batch_chebyshev_distance,

    canberra_distance_kernel,
    canberra_distance,
    instance_canberra_distance,
    batch_canberra_distance,

    exponential_kernel,
    instance_exponential_kernel,
    batch_exponential_kernel,

    gaussian_rbf_kernel,
    instance_gaussian_rbf_kernel,
    batch_gaussian_rbf_kernel,

    laplacian_kernel,
    instance_laplacian_kernel,
    batch_laplacian_kernel,

    anisotropic_rbf_kernel,
    instance_anisotropic_rbf_kernel,
    batch_anisotropic_rbf_kernel,

    custom_hybrid_kernel,
    instance_custom_hybrid_kernel,
    batch_custom_hybrid_kernel
)

from tinybig.koala.linear_algebra.matrix import (
    matrix_power,
    accumulative_matrix_power,
    degree_based_normalize_matrix,
    operator_based_normalize_matrix,
    mean_std_based_normalize_matrix,
    sparse_mx_to_torch_sparse_tensor
)
