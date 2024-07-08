r"""
This module provides the layer, head and component function modules to build the RPN model within the tinyBIG toolkit.

## RPN Model Architecture

Formally, given the underlying data distribution mapping $f: {R}^m \to {R}^n$,
the RPN model proposes to approximate function $f$ as follows:
$$
    \begin{equation}
        g(\mathbf{x} | \mathbf{w}) = \left\langle \kappa(\mathbf{x}), \psi(\mathbf{w}) \right\rangle + \pi(\mathbf{x}).
    \end{equation}
$$

The RPN model disentangles input data from model parameters through the expansion functions $\kappa$ and
reconciliation function $\psi$, subsequently summed with the remainder function $\pi$, where

* $\kappa: {R}^m \to {R}^{D}$ is named as the **data expansion function** (or **data transformation function** to be general) and $D$ is the target expansion space dimension.

* $\psi: {R}^l \to {R}^{n \times D}$ is named as the **parameter reconciliation function** (or **parameter fabrication function** to be general), which is defined only on the parameters without any input data.

* $\pi: {R}^m \to {R}^n$ is named as the **remainder function**.

## RPN Layer with Multi-Head

Similar to the Transformer with multi-head attention, the RPN model employs a multi-head architecture,
where each head can disentangle the input data and model parameters using different expansion, reconciliation
and remainder functions, respectively:
$$
    \begin{equation}
        g(\mathbf{x} | \mathbf{w}, H) = \sum_{h=0}^{H-1} \left\langle \kappa^{(h)}(\mathbf{x}), \psi^{(h)}(\mathbf{w}^{(h)}) \right\rangle + \pi^{(h)}(\mathbf{x}),
    \end{equation}
$$
where the superscript "$h$" indicates the head index and $H$ denotes the total head number.
By default, summation is used to combine the results from all these heads.

## RPN Head with Multi-Channel

Similar to convolutional neural networks (CNNs) employing multiple filters, RPN allows each head to have multiple
channels of parameters applied to the same data expansion.
For example, for the $h_{th}$ head, RPN defines its multi-channel parameters as $\mathbf{w}^{(h),0}, \mathbf{w}^{(h),1}, \cdots, \mathbf{w}^{(h), C-1}$,
where $C$ denotes the number of channels.
These parameters will be reconciled using the same parameter reconciliation function, as shown below:
$$
    \begin{equation}
        g(\mathbf{x} | \mathbf{w}, H, C) = \sum_{h=0}^{H-1} \sum_{c=0}^{C-1} \left\langle \kappa^{(h)}(\mathbf{x}), \psi^{(h)}(\mathbf{w}^{(h), c}) \right\rangle + \pi^{(h)}(\mathbf{x}).
    \end{equation}
$$

## Classes in this Module

This module contains the following categories of component functions and modules:

* Data Transformation/Expansion Function
* Parameter Fabrication/Reconciliation Function
* Remainder Function
* RPN Layer with Multi-Head
* RPN Head with Multi-Channel
"""

from tinybig.module.base_transformation import (
    transformation
)
from tinybig.module.base_fabrication import (
    fabrication
)
from tinybig.module.base_remainder import (
    remainder
)
from tinybig.module.head import (
    rpn_head
)
from tinybig.module.layer import (
    rpn_layer
)
