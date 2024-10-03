# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based MLP Model #
#######################

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import kan_layer


class kan(rpn):
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_kan',
        grid_range=(-1, 1), t: int = 5, d: int = 3,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False, r: int = 3,
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
                kan_layer(
                    m=m, n=n,
                    grid_range=grid_range,
                    t=t, d=d,
                    enable_bias=enable_bias,
                    with_lorr=with_lorr, r=r,
                    device=device,
                    channel_num=channel_num,
                    width=width,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

