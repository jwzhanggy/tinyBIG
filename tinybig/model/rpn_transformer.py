# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based RNN Model #
#######################

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer
from tinybig.layer.bilinear_layers import attention_layer


class transformer(rpn):
    def __init__(
        self,
        dims: list[int] | tuple[int],
        enable_bias: bool = False,
        # optional parameters
        r: int = 3,
        channel_num: int = 1,
        width: int = 1,
        with_batch_norm: bool = True,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        layers = []
        for m, n in zip(dims, dims[1:-2]):
            layers.append(
                attention_layer(
                    m=m, n=n,
                    enable_bias=enable_bias,
                    r=r,
                    channel_num=channel_num,
                    with_batch_norm=with_batch_norm,
                    device=device,
                    *args, **kwargs
                )
            )
            layers.append(
                perceptron_layer(
                    m=n, n=n, device=device,
                    enable_bias=enable_bias,
                    with_batch_norm=with_batch_norm,
                    channel_num=channel_num,
                    width=width,
                )
            )
        m, n = dims[-2:-1]
        layers.append(
            perceptron_layer(
                m=n, n=n, device=device,
                enable_bias=enable_bias,
                with_batch_norm=False,
                with_softmax=True,
                channel_num=channel_num,
                width=width,
            )
        )



