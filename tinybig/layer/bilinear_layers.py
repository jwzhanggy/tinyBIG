# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

############################
# Transformer Layer Module #
############################

from tinybig.module.base_layer import rpn_layer
from tinybig.head.bilinear_heads import bilinear_interdependence_head
from tinybig.fusion.parameterized_concatenation_fusion import parameterized_concatenation_fusion


class bilinear_interdependence_layer(rpn_layer):

    def __init__(
        self,
        m: int, n: int,
        name: str = 'attention_layer',
        batch_num: int = None,
        channel_num: int = 1, width: int = 1,
        # interdependence function parameters
        with_dual_lphm_interdependence: bool = False,
        with_lorr_interdependence: bool = False, r_interdependence: int = 3,
        # data transformation function parameters
        with_taylor: bool = False, d: int = 2,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = False,
        # remainder function parameters
        with_residual: bool = False,
        # output processing parameters
        with_batch_norm: bool = False,
        with_relu: bool = True,
        with_softmax: bool = True,
        with_dropout: bool = False, p: float = 0.25,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        print('* bilinear_interdependence_layer, width:', width)
        heads = [
            bilinear_interdependence_head(
                m=m, n=n,
                batch_num=batch_num,
                channel_num=channel_num,
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
                with_softmax=with_softmax,
                with_dropout=with_dropout, p=p,
                # --------------------------
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        head_fusion = parameterized_concatenation_fusion(
            dims=[n]*width
        )
        print('--------------------------')
        super().__init__(name=name, m=m, n=n, heads=heads, head_fusion=head_fusion, device=device, *args, **kwargs)
