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

## Classes in this Module

This module contains the following categories of RPN models:

* Base Model Template
* Deep RPN Model
* Classic Machine Learning Models: RPN\_SVM, RPN\_PGM, RPN\_Naive\_Bayes
* Neural Network Models: RPN\_MLP, RPN\_KAN
* Vision Models: RPN\_CNN
* Sequential Models: RPN\_RNN, RPN\_Regression\_RNN
* Graph Models: RPN\_GCN, RPN\_GAT
* Transformer Models: RPN\_Transformer
"""

from tinybig.module.base_model import model
from tinybig.model.rpn import rpn
from tinybig.model.rpn_mlp import mlp
from tinybig.model.rpn_kan import kan
from tinybig.model.rpn_pgm import pgm
from tinybig.model.rpn_naive_bayes import naive_bayes
from tinybig.model.rpn_svm import svm
from tinybig.model.rpn_cnn import cnn
from tinybig.model.rpn_gcn import gcn
from tinybig.model.rpn_gat import gat
from tinybig.model.rpn_rnn import rnn
from tinybig.model.rpn_regression_rnn import regression_rnn
from tinybig.model.rpn_transformer import transformer

