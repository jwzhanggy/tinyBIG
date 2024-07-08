r"""
This module provides the remainder functions that can be used to build the RPN model within the tinyBIG toolkit.

## Remainder Function

Formally, to approximate the underlying data distribution mapping $f: {R}^m \to {R}^n$ to be learned,
in addition to the data expansion function and parameter reconciliation function, the remainder function
$\pi$ completes the approximation as a residual term, governing the learning completeness of the RPN model,
which can be represented as follows

$$ \pi: {R}^m \to {R}^{n}.$$

Without specific descriptions, the remainder function $\pi$ defined here is based solely on the input data $\mathbf{x}$.
However, in practice, we also allow $\pi$ to include learnable parameters for output dimension adjustment.
In such cases, it should be rewritten as $\pi(\mathbf{x} | \mathbf{w}')$, where $\mathbf{w}'$ is one extra fraction of the
model's learnable parameters.

## Classes in this Module

This module contains the following categories of remainder functions:

* Basic remainder functions
* Complementary Expansion based functions
"""

from tinybig.module.base_remainder import (
    remainder
)

from tinybig.remainder.basic_remainder import (
    constant_remainder,
    zero_remainder,
    identity_remainder,
    linear_remainder
)

from tinybig.remainder.expansion_remainder import bspline_remainder
