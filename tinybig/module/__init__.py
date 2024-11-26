# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
#     RPN Modules     #
#######################

r"""
This module provides the layer, head and component function modules to build the RPN model within the tinyBIG toolkit.

## RPN Model Architecture

Formally, given the underlying data distribution mapping $f: {R}^m \to {R}^n$,
the RPN model proposes to approximate function $f$ as follows:
$$
    \begin{equation}
        g(\mathbf{x} | \mathbf{w}) = \left \langle \kappa_{\xi} (\mathbf{x}), \psi(\mathbf{w}) \right \rangle + \pi(\mathbf{x}),
    \end{equation}
$$

The RPN model disentangles input data from model parameters through the expansion functions $\kappa$ and
reconciliation function $\psi$, subsequently summed with the remainder function $\pi$, where

* $\kappa_{\xi}: {R}^m \to {R}^{D}$ is named as the **data interdependent transformation function**. It is a composite function of the **data transformation function** $\kappa$ and the **data interdependence function** $\xi$. Notation $D$ is the target expansion space dimension.

* $\psi: {R}^l \to {R}^{n \times D}$ is named as the **parameter reconciliation function** (or **parameter fabrication function** to be general), which is defined only on the parameters without any input data.

* $\pi: {R}^m \to {R}^n$ is named as the **remainder function**.

* $\xi_a: {R}^{b \times m} \to {R}^{m \times m'}$ and $\xi_i: {R}^{b \times m} \to {R}^{b \times b'}$ defined on the input data batch $\mathbf{X} \in R^{b \times m}$ are named as the **attribute** and **instance data interdependence functions**, respectively.

## Data Interdependent Transformation Function

Given this input data batch $\mathbf{X} \in R^{b \times m}$, we can formulate the data interdependence transformation function $\kappa_{\xi}$ as follows:

$$
    \begin{equation}
        \kappa_{\xi}(\mathbf{X}) = \mathbf{A}^\top_{\xi_i} \kappa(\mathbf{X} \mathbf{A}_{\xi_a}) \in {R}^{b' \times D}.
    \end{equation}
$$

These attribute and instance interdependence matrices $\mathbf{A}_{\xi_a} \in {R}^{m \times m'}$ and $\mathbf{A}_{\xi_i} \in {R}^{b \times b'}$ are computed with the corresponding interdependence functions defined above, i.e.,

$$
    \begin{equation}
        \mathbf{A}_{\xi_a} = \xi_a(\mathbf{X}) \in {R}^{m \times m'} \text{, and } \mathbf{A}_{\xi_i} = \xi_i(\mathbf{X}) \in {R}^{b \times b'}.
    \end{equation}
$$

## RPN Layer with Multi-Head

Similar to the Transformer with multi-head attention, the RPN model employs a multi-head architecture,
where each head can disentangle the input data and model parameters using different expansion, reconciliation
and remainder functions, respectively:
$$
    \begin{equation}
        \text{Fusion} \left( \left\\{ \left\langle \kappa^{(h)}_{\xi^{(h)}}(\mathbf{X}), \psi^{(h)}(\mathbf{w}^{(h)}) \right\rangle + \pi^{(h)}(\mathbf{X}) \right\\}; h \in \\{1, 2, \cdots, H \\} \right),
    \end{equation}
$$
where the superscript "$h$" indicates the head index and $H$ denotes the total head number, and $\text{Fusion}(\cdot)$ denotes the multi-head fusion function.
By default, summation is used to combine the results from all these heads.


## RPN Head with Multi-Channel

Similar to convolutional neural networks (CNNs) employing multiple filters, RPN allows each head to have multiple
channels of parameters applied to the same data expansion.
For example, for the $h_{th}$ head, RPN defines its multi-channel parameters as $\mathbf{w}^{(h),0}, \mathbf{w}^{(h),1}, \cdots, \mathbf{w}^{(h), C-1}$,
where $C$ denotes the number of channels.
These parameters will be reconciled using the same parameter reconciliation function, as shown below:
$$
    \begin{equation}
        Fusion \left( \left\\{ \left\langle \kappa^{(h)}_{\xi^{(h), c}}(\mathbf{X}), \psi^{(h)}(\mathbf{w}^{(h), c}) \right\rangle + \pi^{(h)}(\mathbf{X}) \right\\}; h,c \in \\{1, 2, \cdots, H/C \\} \right),
    \end{equation}
$$

## Classes in this Module

This module contains the following categories of component functions and modules:

* Base Function Template
* Data Transformation/Expansion/Compression Functions
* Parameter Fabrication/Reconciliation Functions
* Data/Structural Interdependence Functions
* Remainder Functions
* Fusion Functions
* RPN Layer with Multi-Head
* RPN Head with Multi-Channel
"""

from tinybig.module.base_model import (
    model
)
from tinybig.module.base_function import (
    function
)

from tinybig.module.base_transformation import (
    transformation
)
from tinybig.module.base_interdependence import (
    interdependence
)
from tinybig.module.base_fabrication import (
    fabrication
)
from tinybig.module.base_remainder import (
    remainder
)
from tinybig.module.base_fusion import (
    fusion
)
from tinybig.module.base_head import (
    head,
    head as rpn_head
)
from tinybig.module.base_layer import (
    layer,
    layer as rpn_layer
)

