# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################
# Data Compression Functions #
##############################

r"""
This module provides the "data compression functions" that can be used to build the RPN model within the tinyBIG toolkit.

## Data Compression Function

Different from the "data expansion functions", the data compression functions aim to compress input
data batch from high-dimensional space into lower-dimensional spaces.

Formally, given the underlying data distribution mapping $f: {R}^m \to {R}^n$ to be learned,
the data compression function $\kappa$ projects input data into a new space shown as follows:

$$ \kappa: {R}^m \to {R}^{D}, $$

where the target dimension vector space dimension $D$ is determined when defining $\kappa$.

In practice, the function $\kappa$ can either expand or compress the input to a higher- or lower-dimensional space.
The corresponding function, $\kappa$, can also be referred to as the data expansion function (if $D > m$)
and data compression function (if $D < m$), respectively.
Collectively, these can be unified under the term "data transformation functions".

To differentiate from the dimension notation $D$ of the expansion functions, we will use the
lower-case dimension notation $d$ for compression functions implemented in this module instead.

## Classes in this Module

This module contains the following categories of compression functions:

* Basic compression functions
* Geometric compression functions (based on the cuboid, cylinder and sphere patch shapes)
* Metric based compression functions
* Feature selection based compression functions
* Dimension reduction based compression functions
* Manifold based compression functions
* Probabilistic compression functions
* Extended and Nested compression functions
"""

from tinybig.module.base_transformation import (
    transformation
)

from tinybig.compression.basic_compression import (
    identity_compression,
    reciprocal_compression,
    linear_compression,
)

from tinybig.compression.geometric_compression import (
    geometric_compression,

    cuboid_patch_based_geometric_compression,
    cuboid_mean_based_geometric_compression,
    cuboid_max_based_geometric_compression,
    cuboid_min_based_geometric_compression,

    cylinder_patch_based_geometric_compression,
    cylinder_max_based_geometric_compression,
    cylinder_mean_based_geometric_compression,
    cylinder_min_based_geometric_compression,

    sphere_patch_based_geometric_compression,
    sphere_mean_based_geometric_compression,
    sphere_max_based_geometric_compression,
    sphere_min_based_geometric_compression,
)

from tinybig.compression.metric_based_compression import (
    metric_compression,
    max_compression,
    min_compression,
    mean_compression,
    sum_compression,
    prod_compression,
    median_compression,
)

from tinybig.compression.feature_selection_compression import (
    feature_selection_compression,
    incremental_feature_clustering_based_compression,
    incremental_variance_threshold_based_compression,
)

from tinybig.compression.dimension_reduction_compression import (
    dimension_reduction_compression,
    incremental_PCA_based_compression,
    incremental_random_projection_based_compression,
)

from tinybig.compression.manifold_compression import (
    manifold_compression,
    isomap_manifold_compression,
    lle_manifold_compression,
    mds_manifold_compression,
    tsne_manifold_compression,
    spectral_embedding_manifold_compression,
)

from tinybig.compression.probabilistic_compression import (
    naive_probabilistic_compression,
    naive_uniform_probabilistic_compression,
    naive_chi2_probabilistic_compression,
    naive_laplace_probabilistic_compression,
    naive_cauchy_probabilistic_compression,
    naive_gamma_probabilistic_compression,
    naive_normal_probabilistic_compression,
    naive_exponential_probabilistic_compression,
)

from tinybig.compression.extended_compression import (
    extended_compression,
)
from tinybig.compression.nested_compression import (
    nested_compression,
)
