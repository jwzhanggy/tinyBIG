# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based SVM Model #
#######################

import warnings

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import svm_layer


class svm(rpn):
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_svm',
        kernel: str = 'linear',
        base_range: tuple = (-1, 1),
        num_interval: int = 10,
        epsilon: float = 1.0,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        if len(dims) > 2:
            warnings.warn("Regular SVMs should have only two layers...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                svm_layer(
                    m=m, n=n,
                    enable_bias=enable_bias,
                    kernel=kernel,
                    base_range=base_range,
                    num_interval=num_interval,
                    epsilon=epsilon,
                    with_lorr=with_lorr, r=r,
                    with_residual=with_residual,
                    device=device,
                    channel_num=channel_num,
                    width=width,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

