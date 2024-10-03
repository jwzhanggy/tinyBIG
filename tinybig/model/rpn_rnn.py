# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based RNN Model #
#######################

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer
from tinybig.layer.chain_based_layers import recurrent_layer


class rnn(rpn):
    def __init__(
        self,
        chain_length: int,
        m: int, n: int, d: int,
        hidden_depth: int = 1,
        name: str = 'rpn_rnn',
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

        layers = [
            perceptron_layer(
                m=m, n=d,
                enable_bias=enable_bias,
                with_lorr=with_lorr,
                r=r,
                channel_num=channel_num,
                width=width,
                device=device,
            )
        ]
        for i in range(hidden_depth):
            layers.append(
                recurrent_layer(
                    m=d, n=d,
                    chain_length=chain_length,
                    bi_directional=bi_directional,
                    normalization=normalization,
                    normalization_mode=normalization_mode,
                    self_dependence=self_dependence,
                    require_data=require_data,
                    require_parameters=require_parameters,
                    channel_num=channel_num,
                    with_lorr=with_lorr, r=r,
                    with_residual=with_residual,
                    enable_bias=enable_bias,
                    width=width,
                    device=device, *args, **kwargs
                )
            )
        layers.append(
            perceptron_layer(
                m=d, n=n, device=device,
                enable_bias=enable_bias,
                with_lorr=with_lorr,
                r=r,
                channel_num=channel_num,
                width=width,
            )
        )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

