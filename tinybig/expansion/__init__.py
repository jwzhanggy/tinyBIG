r"""
This module provides the "data expansion functions" that can be used to build the RPN model within the tinyBIG toolkit.

## Data Expansion Function

Formally, given the underlying data distribution mapping $f: {R}^m \to {R}^n$ to be learned,
the data expansion function $\kappa$ projects input data into a new space shown as follows:

$$ \kappa: {R}^m \to {R}^{D}, $$

where the target dimension vector space dimension $D$ is determined when defining $\kappa$.

In practice, the function $\kappa$ can either expand or compress the input to a higher- or lower-dimensional space.
The corresponding function, $\kappa$, can also be referred to as the data expansion function (if $D > m$)
and data compression function (if $D < m$), respectively. Collectively, these can be unified under the term
"data transformation functions".

## Classes in this Module

This module contains the following categories of expansion functions:

* Basic expansion functions
* Polynomial expansion functions
* Trigonometric expansion functions
* RBF expansion functions
* Naive probabilistic expansion functions
* Combinatorial (probabilistic) expansion functions
* Nested expansion function
* Extended expansion function
"""

from tinybig.module.base_transformation import (
    transformation
)

from tinybig.expansion.basic_expansion import (
    identity_expansion,
    reciprocal_expansion,
    linear_expansion,
)

from tinybig.expansion.polynomial_expansion import (
    taylor_expansion,
    fourier_expansion,
)

from tinybig.expansion.trigonometric_expansion import (
    hyperbolic_expansion,
    arc_hyperbolic_expansion,
    trigonometric_expansion,
    arc_trigonometric_expansion,
)

from tinybig.expansion.probabilistic_expansion import (
    normal_expansion,
    cauchy_expansion,
    chi2_expansion,
    gamma_expansion,
    exponential_expansion,
    laplace_expansion,
    hybrid_probabilistic_expansion,
    naive_normal_expansion,
    naive_cauchy_expansion,
    naive_chi2_expansion,
    naive_gamma_expansion,
    naive_exponential_expansion,
    naive_laplace_expansion,
)

from tinybig.expansion.rbf_expansion import (
    gaussian_rbf_expansion,
    inverse_quadratic_rbf_expansion,
)

from tinybig.expansion.recursive_expansion import (
    bspline_expansion,
    chebyshev_expansion,
    jacobi_expansion,
)

from tinybig.expansion.combinatorial_expansion import (
    combinatorial_expansion,
    combinatorial_normal_expansion,
)

from tinybig.expansion.nested_expansion import (
    nested_expansion
)

from tinybig.expansion.extended_expansion import (
    extended_expansion
)

