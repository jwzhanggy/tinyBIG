# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

############################
# Transformer Layer Module #
############################

from tinybig.module.base_layer import rpn_layer
from tinybig.head.bilinear_heads import attention_head
from tinybig.fusion.parameterized_concatenation_fusion import parameterized_concatenation_fusion


class attention_layer(rpn_layer):

    def __init__(
        self,
        m: int, n: int,
        name: str = 'perceptron_layer',
        enable_bias: bool = False,
        # optional parameters
        r: int = 3,
        channel_num: int = 1,
        width: int = 1,
        with_batch_norm: bool = True,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            attention_head(
                m=m, n=n,
                enable_bias=enable_bias,
                r=r,
                channel_num=channel_num,
                with_batch_norm=with_batch_norm,
                device=device,
                *args, **kwargs
            )
        ] * width
        head_fusion = parameterized_concatenation_fusion(
            k=width,
            d=n,
        )
        super().__init__(name=name, m=m, n=n, heads=heads, head_fusion=head_fusion, device=device, *args, **kwargs)
