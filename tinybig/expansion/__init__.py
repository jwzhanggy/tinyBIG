"""
This module provides the expansion functions that can be used within the tinyBIG toolkit.

The module contains the following expansion functions:

the base expansion function
- base_expansion

basic expansion functions
- identity_expansion
- reciprocal_expansion
- linear_expansion

polynomial expansion functions
- taylor_expansion
- fourier_expansion
- bspline_expansion
- chebyshev_expansion
- jacobi_expansion

trigonometric expansion functions
- hyperbolic_expansion
- arc_hyperbolic_expansion
- trigonometric_expansion
- arc_trigonometric_expansion

RBF expansion functions
- gaussian_rbf_expansion
- inverse_quadratic_rbf_expansion

naive probabilistic expansion functions
- normal_expansion
- cauchy_expansion
- chi2_expansion
- gamma_expansion
- exponential_expansion
- laplace_expansion
- hybrid_probabilistic_expansion
- naive_normal_expansion
- naive_cauchy_expansion
- naive_chi2_expansion
- naive_gamma_expansion
- naive_exponential_expansion
- naive_laplace_expansion

combinatorial (probabilistic) expansion functions
- combinatorial_expansion
- combinatorial_normal_expansion

nested and extended expansion functions
- nested_expansion
- extended_expansion
"""

from tinybig.module.transformation import (
    transformation as expansion
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

