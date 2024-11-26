# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
#     RPN Heads     #
#####################

"""
## RPN Layers

In this module, we implement several frequently used RPN layers that can be used to design and construct
the RPN models.

The layers implemented in this module all inherit from the tinybig.module.base_layer.layer class.
For most of the layer, they will only re-define the __init__ method to provide the component functions.
Meanwhile, for some of the layers, they will also implement some other methods to re-define the internal
calculation process, which will be clearly specified in the class implementation.


## Classes in this Module

This module contains the following categories of component layers:

* Basic layers
* Grid based layers
* Chain based layers
* Graph based layers
* Bilinear layers
"""

from tinybig.module.base_layer import (
    layer,
    layer as rpn_layer
)

from tinybig.layer.basic_layers import (
    perceptron_layer,
    svm_layer,
    kan_layer,
    pgm_layer,
    naive_bayes_layer
)

from tinybig.layer.grid_based_layers import (
    grid_interdependence_layer,
    grid_compression_layer
)

from tinybig.layer.chain_based_layers import (
    chain_interdependence_layer,
)

from tinybig.layer.graph_based_layers import (
    graph_interdependence_layer,
    graph_bilinear_interdependence_layer,
)

from tinybig.layer.bilinear_layers import (
    bilinear_interdependence_layer
)

