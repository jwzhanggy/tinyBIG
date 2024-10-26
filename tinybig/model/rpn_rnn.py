# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based RNN Model #
#######################
import torch

import tinybig.layer
from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer
from tinybig.layer.chain_based_layers import chain_interdependence_layer


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
        self_dependence: bool = True,
        self_scaling: float = 1.0,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        with_residual: bool = False,
        # bilinear interdependence function parameters
        with_dual_lphm_interdependence: bool = False,
        with_lorr_interdependence: bool = False, r_interdependence: int = 3,
        # remainder function parameters
        enable_bias: bool = False,
        # output processing parameters
        with_batch_norm: bool = False,
        with_relu: bool = True,
        with_softmax: bool = True,
        with_dropout: bool = False, p: float = 0.25,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        print('############# rpn-rnn model architecture ############')

        self.chain_length = chain_length

        if dims is None or len(dims) <= 1:
           raise ValueError('dims must not be empty and need to have at least two dimensions...')
        assert all(isinstance(d, int) and d > 0 for d in dims)

        # input embedding layer
        layers = []
        for m, n in zip(dims[0:-2], dims[1:-1]):
            print('m', m, 'n', n)
            #---------------- x to h -----------------
            layers.append(
                perceptron_layer(
                    m=m, n=n,
                    channel_num=channel_num,
                    width=width,
                    # -----------------------
                    with_dual_lphm=with_dual_lphm,
                    with_lorr=with_lorr, r=r,
                    enable_bias=enable_bias,
                    with_residual=with_residual,
                    # -----------------------
                    with_batch_norm=False,
                    with_relu=True,
                    with_softmax=False,
                    with_dropout=False, p=p,
                    # -----------------------
                    parameters_init_method=parameters_init_method,
                    device=device,
                )
            )
            # ---------------- h to h -----------------
            layers.append(
                chain_interdependence_layer(
                    m=n, n=n,
                    chain_length=chain_length,
                    channel_num=channel_num,
                    width=width,
                    # -----------------------
                    bi_directional=bi_directional,
                    with_multihop=with_multihop, h=h, accumulative=accumulative,
                    with_inverse_approx=with_inverse_approx,
                    with_exponential_approx=with_exponential_approx,
                    self_dependence=self_dependence,
                    self_scaling=self_scaling,
                    # -----------------------
                    with_dual_lphm=with_dual_lphm,
                    with_lorr=with_lorr, r=r,
                    with_residual=with_residual,
                    # -----------------------
                    with_dual_lphm_interdependence=with_dual_lphm_interdependence,
                    with_lorr_interdependence=with_lorr_interdependence,
                    r_interdependence=r_interdependence,
                    # -----------------------
                    enable_bias=enable_bias,
                    # -----------------------
                    with_batch_norm=with_batch_norm,
                    with_relu=with_relu,
                    with_softmax=False,
                    with_dropout=with_dropout, p=p,
                    # -----------------------
                    parameters_init_method=parameters_init_method,
                    device=device,
                )
            )
        #--------------- output layer: h to y ------------------
        layers.append(
            perceptron_layer(
                name='output_layer',
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
                parameters_init_method=parameters_init_method,
                device=device,
            )
        )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        for layer in self.layers:
            if isinstance(layer, tinybig.layer.perceptron_layer):
                if layer.name is not None and layer.name == 'output_layer':
                    x = x.view(x.size(0), self.chain_length, -1)
                    x = x.mean(dim=1)
                    x = layer(x, device=device)
                else:
                    b, m = x.shape
                    x = x.view(b * self.chain_length, -1)
                    x = layer(x, device=device)
                    x = x.view(b, -1)
            else:
                x = layer(x, device=device)
        return x