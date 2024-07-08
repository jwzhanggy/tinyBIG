r"""
This module provides the parameter reconciliation functions that can be used to build the RPN model within the tinyBIG toolkit.

## Parameter Reconciliation Function

Formally, given the underlying data distribution mapping $f: {R}^m \to {R}^n$ to be learned,
the parameter reconciliation function $\psi$ adjusts the available parameter vector of length $l$ by fabricating
a new parameter matrix of size $n \times D$ to accommodate the expansion space dimension $D$ as follows:

$$ \psi: {R}^l \to {R}^{n \times D}, $$

which is defined only on the parameters without any input data.

In most of the cases, the parameter vector length $l$ is much smaller than the output matrix size $n \times D$,
i.e., $l \ll n \times D$.
Meanwhile, in practice, we can also define function $\psi$ to fabricate a longer parameter vector into a smaller
parameter matrix, i.e., $l > n \times D$.
To unify these different cases, the data reconciliation function can also be referred to as the
"parameter fabrication function", and these function names will be used interchangeably.

## Classes in this Module

This module contains the following categories of parameter reconciliation functions:

* Basic reconciliation functions
* Lowrank reconciliation functions
* Hypernet reconciliation functions
"""

from tinybig.module.base_fabrication import (
    fabrication
)

from tinybig.reconciliation.basic_reconciliation import (
    constant_reconciliation,
    one_reconciliation,
    zero_reconciliation,
    constant_eye_reconciliation,
    identity_reconciliation,
    masking_reconciliation,
    duplicated_padding_reconciliation
)

from tinybig.reconciliation.lowrank_reconciliation import (
    lorr_reconciliation,
    hm_reconciliation,
    lphm_reconciliation,
    dual_lphm_reconciliation,
)

from tinybig.reconciliation.hypernet_reconciliation import hypernet_reconciliation
