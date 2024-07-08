r"""
This tinybig library implements the RPN functions, components, modules and models that can be used to address
various deep function learning tasks.

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

## Deep RPN model with Multi-Layer

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

## Learning Correctness of RPN: Complexity, Capacity and Completeness

The **learning correctness** of RPN is fundamentally determined by the compositions of its component functions, each contributing from different perspectives:

* **Model Complexity**: The data expansion function $\kappa$ expands the input data by projecting its representations using basis vectors in the new space. In other words, function $\kappa$ determines the upper bound of the RPN model's complexity.

* **Model Capacity**: The reconciliation function $\psi$ processes the parameters to match the dimensions of the expanded data vectors. The reconciliation function and parameters jointly determine the learning capacity and associated training costs of the RPN model.

* **Model Completeness**: The remainder function $\pi$ completes the approximation as a residual term, governing the learning completeness of the RPN model.

## Learning Cost of RPN: Space, Time and Parameter Number

To analyze the learning costs of RPN, we can take a batch input $\mathbf{X} \in \mathbbm{R}^{B \times m}$ of batch size $B$ as an example, which will be fed to the RPN model with $K$ layers, each with $H$ heads and each head has $C$ channels. Each head will project the data instance from a vector of length $m$ to an expanded vector of length $D$ and then further projected to the desired output of length $n$. Each channel reconciles parameters from length $l$ to the sizes determined by both the expansion space and output space dimensions, {\ie} $n \times D$.

Based on the above hyper-parameters, assuming the input and output dimensions at each layer are comparable to $m$ and $n$, then the space, time costs and the number of involved parameters in learning the RPN model are calculated as follows:

* **Space Cost**: The total space cost for data (including the inputs, expansions and outputs) and parameter (including raw parameters, fabricated parameters generated by the reconciliation function and optional remainder function parameters) can be represented as $\mathcal{O}( K H (B (m + D + n )  + C (l + nD) + mn))$.

* **Time Cost**: Depending on the expansion and reconciliation functions used for building RPN, the total time cost of RPN can be represented as $\mathcal{O}( K H (t_{exp}(m, D) + C t_{rec}(l, D) + C m n D + m n))$, where notations $t_{exp}(m, D)$ and $t_{rec}(l, D)$ denote the expected time costs for data expansion and parameter reconciliation functions, respectively.

* **Learnable parameters**: The total number of parameters in RPN will be $\mathcal{O}(K H C l + K H m n)$, where $\mathcal{O}( K H m n)$ denotes the optional parameter number used for defining the remainder function.

"""


__version__ = '0.1.0.post9'

from . import model, module, config
from . import remainder, expansion, compression, reconciliation
from . import learner, data, output, metric, koala
from . import visual, util

__all__ = [
    'model',
    'module',
    'config',
    'remainder',
    'expansion',
    'compression',
    'reconciliation',
    'learner',
    'data',
    'output',
    'metric',
    'koala',
    'visual',
    'util'
]
