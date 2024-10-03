# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# RPN based Naive Bayes Model #
###############################

import warnings

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import naive_bayes_layer


class naive_bayes(rpn):
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_naive_bayes',
        distribution: str = 'normal',
        enable_bias: bool = False,
        device: str = 'cpu',
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        *args, **kwargs
    ):
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        if len(dims) > 2:
            warnings.warn("Regular SVMs should have only two layers...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                naive_bayes_layer(
                    m=m, n=n, device=device,
                    enable_bias=enable_bias,
                    distribution=distribution,
                    with_lorr=with_lorr,
                    with_residual=with_residual,
                    r=r,
                    channel_num=channel_num,
                    width=width,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

