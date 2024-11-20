# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
#     RPN Heads     #
#####################


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
