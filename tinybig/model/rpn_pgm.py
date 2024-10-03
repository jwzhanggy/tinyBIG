# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################
# RPN based Bayesian Network #
##############################

import warnings

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import pgm_layer


class pgm(rpn):
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_bayesian_network',
        distribution: str = 'normal',
        d: int = 2, with_replacement: bool = False,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device='cpu', *args, **kwargs
    ):
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        if len(dims) > 2:
            warnings.warn("Regular SVMs should have only two layers...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                pgm_layer(
                    m=m, n=n,
                    enable_bias=enable_bias,
                    distribution=distribution,
                    d=d, with_replacement=with_replacement,
                    with_lorr=with_lorr, r=r,
                    with_residual=with_residual,
                    channel_num=channel_num,
                    width=width,
                    device=device, *args, **kwargs
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

