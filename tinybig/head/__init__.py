# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
#     RPN Heads     #
#####################

"""
## RPN Heads

In this module, we implement several frequently used RPN head that can be used to design and construct
the RPN based function learning models.

The heads implemented in this module all inherit from the tinybig.module.base_head.head class.
For most of the heads, they will only re-define the __init__ method to provide the component functions.
Meanwhile, for some of the heads, they will also implement some other methods to re-define the internal
calculation process, which will be clearly specified in the class implementation.


## Classes in this Module

This module contains the following categories of component heads:

* Basic heads
* Grid based heads
* Chain based heads
* Graph based heads
* Bilinear heads
"""

from tinybig.module.base_head import (
    head,
    head as rpn_head
)

from tinybig.head.basic_heads import (
    perceptron_head,
    svm_head,
    kan_head,
    pgm_head,
    naive_bayes_head,
)

from tinybig.head.grid_based_heads import (
    grid_interdependence_head,
    grid_compression_head,
)

from tinybig.head.chain_based_heads import (
    chain_interdependence_head,
)

from tinybig.head.graph_based_heads import (
    graph_interdependence_head,
)

from tinybig.head.bilinear_heads import (
    bilinear_interdependence_head
)
