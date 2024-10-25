# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based RNN Model #
#######################

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer
from tinybig.layer.bilinear_layers import bilinear_interdependence_layer


class transformer(rpn):
    def __init__(
        self,
        dims: list[int] | tuple[int],
        fc_dims: list[int] | tuple[int],
        batch_num: int = None,
        name: str = 'rpn_transformer',
        channel_num: int = 1, width: int = 1,
        # interdependence function parameters
        with_dual_lphm_interdependence: bool = False,
        with_lorr_interdependence: bool = True, r_interdependence: int = 3,
        # data transformation function parameters
        with_taylor: bool = False, d: int = 2,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = True,
        # remainder function parameters
        with_residual: bool = True,
        # output processing parameters
        with_batch_norm: bool = True,
        with_relu: bool = True,
        with_softmax: bool = True,
        with_dropout: bool = True, p: float = 0.25,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        print('############# rpn-transformer model architecture ############')

        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")
        if len(fc_dims) < 1:
            raise ValueError("At least one fc_dim value is needed for defining the model...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                bilinear_interdependence_layer(
                    m=m, n=n,
                    batch_num=batch_num,
                    channel_num=channel_num,
                    width=width,
                    # --------------------------
                    with_dual_lphm_interdependence=with_dual_lphm_interdependence,
                    with_lorr_interdependence=with_lorr_interdependence, r_interdependence=r_interdependence,
                    # --------------------------
                    with_taylor=with_taylor, d=d,
                    # --------------------------
                    with_dual_lphm=with_dual_lphm,
                    with_lorr=with_lorr, r=r,
                    enable_bias=enable_bias,
                    # --------------------------
                    with_residual=with_residual,
                    # --------------------------
                    with_batch_norm=with_batch_norm,
                    with_relu=with_relu,
                    with_dropout=with_dropout, p=p,
                    with_softmax=with_softmax,
                    # --------------------------
                    device=device,
                )
            )
            layers.append(
                perceptron_layer(
                    m=n, n=n,
                    channel_num=channel_num,
                    width=width,
                    # --------------------------
                    with_taylor=with_taylor, d=d,
                    # --------------------------
                    with_dual_lphm=with_dual_lphm,
                    with_lorr=with_lorr, r=r,
                    enable_bias=enable_bias,
                    # --------------------------
                    with_residual=with_residual,
                    # --------------------------
                    with_batch_norm=with_batch_norm,
                    with_relu=with_relu,
                    with_dropout=with_dropout, p=p,
                    with_softmax=with_softmax,
                    # --------------------------
                    device=device,
                )
            )
        fc_dims = [dims[-1]] + list(fc_dims)
        for m, n in zip(fc_dims, fc_dims[1:]):
            layers.append(
                perceptron_layer(
                    m=m, n=n,
                    channel_num=channel_num,
                    width=width,
                    # --------------------------
                    with_taylor=with_taylor, d=d,
                    # --------------------------
                    with_dual_lphm=with_dual_lphm,
                    with_lorr=with_lorr, r=r,
                    enable_bias=enable_bias,
                    # --------------------------
                    with_residual=with_residual,
                    # --------------------------
                    with_batch_norm=with_batch_norm and n != fc_dims[-1],
                    with_relu=with_relu and n != fc_dims[-1],
                    with_dropout=with_dropout and n != fc_dims[-1], p=p,
                    with_softmax=with_softmax and m == fc_dims[-2] and n == fc_dims[-1],
                    # --------------------------
                    device=device,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)


