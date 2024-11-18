r"""
This tinybig library implements the RPN functions, components, modules and models that can be used to address
various deep function learning tasks.

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

## Deep RPN model with Multi-Layer

The multi-head multi-channel RPN layer provides RPN with greater capabilities
for approximating functions with diverse expansions concurrently.
However, such shallow architectures can be insufficient for modeling complex functions.
The RPN model can also be designed with a deep architecture by stacking multiple RPN layers on top of each other.

Formally, we can represent the deep RPN model with multi-layers as follows:

$$
    \begin{equation}
        \begin{cases}
            \text{Input: } & \mathbf{H}_0  = \mathbf{X},\\\\
            \text{Layer 1: } & \mathbf{H}_1 = \left\langle \kappa_{\xi, 1}(\mathbf{H}_0), \psi_1(\mathbf{w}_1) \right\rangle + \pi_1(\mathbf{H}_0),\\\\
            \text{Layer 2: } & \mathbf{H}_2 = \left\langle \kappa_{\xi, 2}(\mathbf{H}_1), \psi_2(\mathbf{w}_2) \right\rangle + \pi_2(\mathbf{H}_1),\\\\
            \cdots & \cdots \ \cdots\\\\
            \text{Layer K: } & \mathbf{H}_K = \left\langle \kappa_{\xi, K}(\mathbf{H}_{K-1}), \psi_K(\mathbf{w}_K) \right\rangle + \pi_K(\mathbf{H}_{K-1}),\\\\
            \text{Output: } & \mathbf{Z}  = \mathbf{H}_K.
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

"""


__version__ = '0.2.0'

from . import model, application
from . import module, head, layer, config
from . import expansion, compression, transformation, reconciliation, remainder, interdependence, fusion
from . import koala
from . import data, output
from . import loss, metric, learner, optimizer
from . import visual, util


__all__ = [
    # ---- models and applications ----
    'application',
    'model',
    # ---- modules ----
    'module',
    'config',
    'head',
    'layer',
    # ---- component functions ----
    'expansion',
    'compression',
    'transformation',
    'reconciliation',
    'remainder',
    'interdependence',
    'fusion',
    # ---- other libraries ----
    'koala',
    # ---- input and output ----
    'data',
    'output',
    # ---- learning ----
    'loss',
    'metric',
    'learner',
    'optimizer',
    # ---- visualization and utility ----
    'visual',
    'util'
]
