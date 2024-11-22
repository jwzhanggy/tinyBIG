# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Data Interdependence Functions #
##################################

r"""

This module provides the "data interdependence functions" that can be used to build the RPN model within the tinyBIG toolkit.

## Data Interdependence Function

The data interdependence functions $\xi$ capture the intricate interdependence relationships among data instances and attributes.
These functions will extract nuanced information from the input data batch, operating both prior to and following the data
projection facilitated by function $\kappa$.

Formally, given an input data batch $\mathbf{X} \in {R}^{b \times m}$ (with $b$ instances and each instance with $m$ attributes),
the attribute and instance data interdependence functions are defined as:

$$
    \begin{equation}
    \xi_a: {R}^{b \times m} \to {R}^{m \times m'} \text{, and }
    \xi_i: {R}^{b \times m} \to {R}^{b \times b'},
    \end{equation}
$$

where $m'$ and $b'$ denote the output dimensions of their respective interdependence functions, respectively.

## Data Interdependent Transformation Function

To elucidate the mechanisms of attribute and instance interdependence functions in defining the data interdependence transformation function $\kappa_{\xi}$,
we shall consider a multi-instance input data batch $\mathbf{X} \in {R}^{b \times m}$ as an exemplar.
Here, $b$ and $m$ denote the number of instances and attributes, respectively.
Given this input data batch $\mathbf{X}$, we can formulate the data interdependence transformation function $\kappa_{\xi}$ as follows:

$$
    \begin{equation}
    \kappa_{\xi}(\mathbf{X}) = \mathbf{A}^\top_{\xi_i} \kappa(\mathbf{X} \mathbf{A}_{\xi_a}) \in {R}^{b' \times D}.
    \end{equation}
$$

These attribute and instance interdependence matrices $\mathbf{A}_{\xi_a} \in {R}^{m \times m'}$ and $\mathbf{A}_{\xi_i} \in {R}^{b \times b'}$
are computed with the corresponding interdependence functions defined above, i.e.,

$$
    \begin{equation}
    \mathbf{A}_{\xi_a} = \xi_a(\mathbf{X}) \in {R}^{m \times m'} \text{, and } \mathbf{A}_{\xi_i} = \xi_i(\mathbf{X}) \in {R}^{b \times b'}.
    \end{equation}
$$

The dimension of the target transformation space, denoted as $D$, is determined by the codomain dimension $m'$ of the attribute interdependence function.
In most cases, the domain and codomain dimensions of the attribute and instance dependence functions analyzed in this paper are identical, i.e., $m' = m$ and $b' = b$.

## Classes in this Module

This module contains the following categories of compression functions:

* Basic interdependence functions
* Geometric interdependence functions (based on the cuboid, cylinder and sphere patch shapes)
* Topological interdependence functions (based on graph and chain structures)
* Kernel based interdependence functions
* Parameterized interdependence functions
* Parameterized bilinear interdependence functions
* Parameterized RPN based interdependence functions
* Hybrid interdependence functions
"""


from tinybig.module.base_interdependence import (
    interdependence
)

from tinybig.interdependence.basic_interdependence import (
    constant_interdependence,
    constant_c_interdependence,
    zero_interdependence,
    one_interdependence,
    identity_interdependence,
    identity_interdependence as eye_interdependence,
)

from tinybig.interdependence.statistical_kernel_interdependence import (
    statistical_kernel_based_interdependence,
    kl_divergence_interdependence,
    pearson_correlation_interdependence,
    rv_coefficient_interdependence,
    mutual_information_interdependence
)

from tinybig.interdependence.numerical_kernel_interdependence import (
    numerical_kernel_based_interdependence,
    linear_kernel_interdependence,
    polynomial_kernel_interdependence,
    hyperbolic_tangent_kernel_interdependence,
    exponential_kernel_interdependence,
    minkowski_distance_interdependence,
    manhattan_distance_interdependence,
    euclidean_distance_interdependence,
    chebyshev_distance_interdependence,
    canberra_distance_interdependence,
    cosine_similarity_interdependence,
    gaussian_rbf_kernel_interdependence,
    laplacian_kernel_interdependence,
    anisotropic_rbf_kernel_interdependence,
    custom_hybrid_kernel_interdependence,
)

from tinybig.interdependence.parameterized_interdependence import (
    parameterized_interdependence,
    lowrank_parameterized_interdependence,
    hm_parameterized_interdependence,
    lphm_parameterized_interdependence,
    dual_lphm_parameterized_interdependence,
    random_matrix_adaption_parameterized_interdependence
)

from tinybig.interdependence.parameterized_bilinear_interdependence import (
    parameterized_bilinear_interdependence,
    lowrank_parameterized_bilinear_interdependence,
    hm_parameterized_bilinear_interdependence,
    lphm_parameterized_bilinear_interdependence,
    dual_lphm_parameterized_bilinear_interdependence,
    random_matrix_adaption_parameterized_bilinear_interdependence
)

from tinybig.interdependence.topological_interdependence import (
    graph_interdependence,
    multihop_graph_interdependence,
    pagerank_multihop_graph_interdependence,
    chain_interdependence,
    multihop_chain_interdependence,
    inverse_approx_multihop_chain_interdependence,
    exponential_approx_multihop_chain_interdependence
)

from tinybig.interdependence.geometric_interdependence import (
    geometric_interdependence,
    cuboid_patch_based_geometric_interdependence,
    cylinder_patch_based_geometric_interdependence,
    sphere_patch_based_geometric_interdependence,
    cuboid_patch_padding_based_geometric_interdependence,
    cuboid_patch_aggregation_based_geometric_interdependence,
    cylinder_patch_padding_based_geometric_interdependence,
    cylinder_patch_aggregation_based_geometric_interdependence,
    sphere_patch_padding_based_geometric_interdependence,
    sphere_patch_aggregation_based_geometric_interdependence,
)

from tinybig.interdependence.parameterized_rpn_interdependence import (
    parameterized_rpn_interdependence
)

from tinybig.interdependence.hybrid_interdependence import (
    hybrid_interdependence
)