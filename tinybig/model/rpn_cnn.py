# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based CNN Model #
#######################

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer
from tinybig.layer.grid_based_layers import conv_layer, pooling_layer


class cnn(rpn):
    def __init__(
        self,
        h: int, w: int,
        channel_nums: list[int] | tuple[int],
        fc_dims: list[int] | tuple[int],
        d: int = 1,
        fc_channel_num: int = 1,
        width: int = 1,
        pooling_metric: str = 'batch_max',
        pooling_layer_gaps: int = 2,
        name: str = 'rpn_cnn',
        # patch structure parameters
        patch_shape: str = 'cuboid',
        p_h: int = None, p_h_prime: int = None,
        p_w: int = None, p_w_prime: int = None,
        p_d: int = 0, p_d_prime: int = None,
        p_r: int = None,
        # packing parameters
        cd_h: int = None, cd_w: int = None, cd_d: int = 1,
        packing_strategy: str = 'densest_packing',
        pooling_cd_h: int = None, pooling_cd_w: int = None, pooling_cd_d: int = 1,
        # output processing function parameters
        with_batch_norm: bool = True,
        with_relu: bool = True,
        with_softmax: bool = False,
        with_residual: bool = False,
        with_dropout: bool = True,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = True,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if pooling_cd_h is None: pooling_cd_h = cd_h
        if pooling_cd_w is None: pooling_cd_w = cd_w
        if pooling_cd_d is None: pooling_cd_d = cd_d

        layers = []
        for in_channel, out_channel in zip(channel_nums, channel_nums[1:]):
            print('conv in', h, w, d, in_channel)
            layer = conv_layer(
                h=h, w=w, d=d,
                in_channel=in_channel, out_channel=out_channel,
                width=width,
                patch_shape=patch_shape,
                p_h=p_h, p_h_prime=p_h_prime,
                p_w=p_w, p_w_prime=p_w_prime,
                p_d=p_d, p_d_prime=p_d_prime,
                p_r=p_r,
                cd_h=cd_h, cd_w=cd_w, cd_d=cd_d,
                packing_strategy=packing_strategy,
                with_batch_norm=with_batch_norm,
                with_relu=with_relu,
                with_residual=with_residual,
                enable_bias=enable_bias,
                with_dual_lphm=with_dual_lphm,
                with_lorr=with_lorr, r=r,
                device=device, *args, **kwargs
            )
            h, w, d = layer.get_output_grid_shape()
            print('conv out', h, w, d, out_channel)
            layers.append(layer)

            # adding a pooling layer for a certain layer gaps
            if len(layers) % (pooling_layer_gaps+1) == pooling_layer_gaps:
                print('pooling in', h, w, d, out_channel)
                layer = pooling_layer(
                    h=h, w=w, d=d,
                    channel_num=out_channel,
                    pooling_metric=pooling_metric,
                    patch_shape=patch_shape,
                    p_h=1, p_h_prime=0,
                    p_w=1, p_w_prime=0,
                    p_d=0, p_d_prime=0,
                    p_r=p_r,
                    cd_h=pooling_cd_h, cd_w=pooling_cd_w, cd_d=pooling_cd_d,
                    packing_strategy=packing_strategy,
                    with_dropout=with_dropout,
                    device=device, *args, **kwargs
                )
                h, w, d = layer.get_output_grid_shape()
                print('pooling out', h, w, d, out_channel)
                layers.append(layer)
                p_h = max(int(p_h/2), 1)
                p_w = max(int(p_w/2), 1)

        # perceptron layers
        assert len(layers) >= 1
        m = layers[-1].get_n()
        dims = [m] + fc_dims
        for m, n in zip(dims, dims[1:]):
            print('fc in', m)
            layers.append(
                perceptron_layer(
                    m=m, n=n,
                    enable_bias=enable_bias,
                    with_dual_lphm=with_dual_lphm,
                    with_lorr=with_lorr, r=r,
                    with_residual=with_residual,
                    channel_num=fc_channel_num,
                    width=width,
                    with_batch_norm=with_batch_norm and n != dims[-1],
                    with_relu=with_relu and n != dims[-1],
                    with_dropout=with_dropout and n != dims[-1],
                    with_softmax=with_softmax and m == dims[-2] and n == dims[-1],
                    device=device, *args, **kwargs
                )
            )
            print('fc out', n)
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)


class resnet(cnn):
    def __init__(self, with_residual: bool = True, *args, **kwargs):
        super().__init__(with_residual=True, *args, **kwargs)