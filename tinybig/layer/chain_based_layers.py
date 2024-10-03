# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################
# Chain based RPN Layer Module #
################################

from tinybig.module.base_layer import rpn_layer
from tinybig.head.chain_based_heads import recurrent_head


class recurrent_layer(rpn_layer):
    def __init__(
        self,
        m: int, n: int,
        chain_length: int,
        name: str = 'recurrent_layer',
        bi_directional: bool = False,
        normalization: bool = False,
        normalization_mode: str = 'row_column',
        self_dependence: bool = False,
        require_data: bool = False,
        require_parameters: bool = False,
        channel_num: int = 1,
        with_lorr: bool = False, r: int = 3,
        with_residual: bool = True,
        enable_bias: bool = False,
        width: int = 1,
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            recurrent_head(
                m=m, n=n,
                chain_length=chain_length,
                bi_directional=bi_directional,
                normalization=normalization,
                normalization_mode=normalization_mode,
                self_dependence=self_dependence,
                require_data=require_data,
                require_parameters=require_parameters,
                channel_num=channel_num,
                with_lorr=with_lorr,
                r=r,
                with_residual=with_residual,
                enable_bias=enable_bias,
                device=device,
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)
