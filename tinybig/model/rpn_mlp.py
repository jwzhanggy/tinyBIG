# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based MLP Model #
#######################

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer


class mlp(rpn):
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_mlp',
        enable_bias: bool = False,
        # optional parameters
        with_taylor: bool = False,
        d: int = 2,
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

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                perceptron_layer(
                    m=m, n=n, device=device,
                    enable_bias=enable_bias,
                    with_taylor=with_taylor,
                    with_lorr=with_lorr,
                    with_residual=with_residual,
                    d=d, r=r,
                    channel_num=channel_num,
                    width=width,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

