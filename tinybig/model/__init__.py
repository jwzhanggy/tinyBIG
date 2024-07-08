r"""
This module provides the deep RPN model build by stacking multiple RPN layers on top of each other.

## Deep RPN model

The multi-head multi-channel RPN layer provides RPN with greater capabilities
for approximating functions with diverse expansions concurrently.
However, such shallow architectures can be insufficient for modeling complex functions.
The RPN model can also be designed with a deep architecture by stacking multiple RPN layers on top of each other.

Formally, we can represent the deep RPN model with multi-layers as follows:

$$
    \begin{equation}
        \begin{cases}
            \text{Input: } & \mathbf{h}_0  = \mathbf{x},\\\\
            \text{Layer 1: } & \mathbf{h}_1 = \left\langle \kappa_1(\mathbf{h}_0), \psi_1(\mathbf{w}_1) \right\rangle + \pi_1(\mathbf{h}_0),\\\\
            \text{Layer 2: } & \mathbf{h}_2 = \left\langle \kappa_2(\mathbf{h}_1), \psi_2(\mathbf{w}_2) \right\rangle + \pi_2(\mathbf{h}_1),\\\\
            \cdots & \cdots \ \cdots\\\\
            \text{Layer K: } & \mathbf{h}_K = \left\langle \kappa_K(\mathbf{h}_{K-1}), \psi_K(\mathbf{w}_K) \right\rangle + \pi_K(\mathbf{h}_{K-1}),\\\\
            \text{Output: } & \hat{\mathbf{y}}  = \mathbf{h}_K.
        \end{cases}
    \end{equation}
$$

In the above equation, the subscripts used above denote the layer index. The dimensions of the outputs at each layer
can be represented as a list $[d_0, d_1, \cdots, d_{K-1}, d_K]$, where $d_0 = m$ and $d_K = n$
denote the input and the desired output dimensions, respectively.
Therefore, if the component functions at each layer of our model have been predetermined, we can just use the dimension 
list $[d_0, d_1, \cdots, d_{K-1}, d_K]$ to represent the architecture of the RPN model.

## Classes in this Module

This module contains the following categories of RPN models:

* Base Model Template
* Deep RPN Model with Multiple RPN-Layers.
"""

from .base_model import model
from .rpn import rpn
