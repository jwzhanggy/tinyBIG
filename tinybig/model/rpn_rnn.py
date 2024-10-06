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
        dims: list[int] | tuple[int],
        name: str = 'rpn_rnn',
        channel_num: int = 1,
        width: int = 1,
        # chain structure interdependence function parameters
        bi_directional: bool = False,
        with_multihop: bool = False, h: int = 1, accumulative: bool = False,
        with_inverse_approx: bool = False,
        with_exponential_approx: bool = False,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        with_residual: bool = False,
        # remainder function parameters
        enable_bias: bool = False,
        # output processing parameters
        with_batch_norm: bool = False,
        with_relu: bool = True,
        with_softmax: bool = True,
        with_dropout: bool = False, p: float = 0.25,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if dims is None or len(dims) < 2:
            raise ValueError('dims must not be empty and need to have at least two dimensions...')
        assert all(isinstance(d, int) and d > 0 for d in dims)

        # input embedding layer
        layers = [
            perceptron_layer(
                m=dims[0], n=dims[1],
                channel_num=channel_num,
                width=width,
                # -----------------------
                with_dual_lphm=with_dual_lphm,
                with_lorr=with_lorr, r=r,
                enable_bias=enable_bias,
                with_residual=with_residual,
                # -----------------------
                with_batch_norm=with_batch_norm and len(dims) != 2,
                with_relu=with_relu and len(dims) != 2,
                with_softmax=with_softmax and len(dims) == 2,
                with_dropout=with_dropout and len(dims) != 2, p=p,
                # -----------------------
                device=device,
            )
        ]
        if len(dims) > 2:
            for m, n in zip(dims[1:-2], dims[2:-1]):
                layers.append(
                    recurrent_layer(
                        m=m, n=n,
                        chain_length=chain_length,
                        channel_num=channel_num,
                        width=width,
                        # -----------------------
                        bi_directional=bi_directional,
                        with_multihop=with_multihop, h=h, accumulative=accumulative,
                        with_inverse_approx=with_inverse_approx,
                        with_exponential_approx=with_exponential_approx,
                        # -----------------------
                        with_dual_lphm=with_dual_lphm,
                        with_lorr=with_lorr, r=r,
                        with_residual=with_residual,
                        # -----------------------
                        enable_bias=enable_bias,
                        # -----------------------
                        with_batch_norm=with_batch_norm,
                        with_relu=with_relu,
                        with_softmax=False,
                        with_dropout=with_dropout, p=p,
                        # -----------------------
                        device=device,
                    )
                )
            layers.append(
                perceptron_layer(
                    m=dims[-2], n=dims[-1],
                    channel_num=channel_num,
                    width=width,
                    # -----------------------
                    with_dual_lphm=with_dual_lphm,
                    with_lorr=with_lorr, r=r,
                    enable_bias=enable_bias,
                    with_residual=with_residual,
                    # -----------------------
                    with_batch_norm=False,
                    with_relu=False,
                    with_softmax=with_softmax,
                    with_dropout=False, p=p,
                    # -----------------------
                    device=device,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

