# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################
# Data Compression Functions #
##############################


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
