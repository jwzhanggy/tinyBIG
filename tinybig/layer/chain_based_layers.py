# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################
# Chain based RPN Layer Module #
################################
import torch

from tinybig.module.base_layer import rpn_layer
from tinybig.head.chain_based_heads import chain_interdependence_head


class chain_interdependence_layer(rpn_layer):
    def __init__(
        self,
        m: int, n: int,
        chain_length: int,
        channel_num: int = 1,
        width: int = 1,
        name: str = 'chain_interdependence_layer',
        # interdependence function parameters
        bi_directional: bool = False,
        with_multihop: bool = False, h: int = 1, accumulative: bool = False,
        with_inverse_approx: bool = False,
        with_exponential_approx: bool = False,
        self_dependence: bool = True,
        self_scaling: float = 1.0,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = False,
        # remainder function parameters
        with_residual: bool = False,
        # output processing parameters
        with_batch_norm: bool = False,
        with_relu: bool = True,
        with_dropout: bool = False, p: float = 0.25,
        with_softmax: bool = True,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, ** kwargs
    ):
        print('* chain_interdependence_layer, width:', width)
        heads = [
            chain_interdependence_head(
                m=m, n=n,
                chain_length=chain_length,
                channel_num=channel_num,
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
                enable_bias=enable_bias,
                # -----------------------
                with_residual=with_residual,
                # -----------------------
                with_batch_norm=with_batch_norm,
                with_relu=with_relu,
                with_dropout=with_dropout, p=p,
                with_softmax=with_softmax,
                # -----------------------
                parameters_init_method=parameters_init_method,
                device=device, *args, ** kwargs
            )
        ] * width
        print('--------------------------')
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


    def forward(self, x: torch.Tensor, fusion_strategy: str = 'average', device: str = 'cpu', *args, **kwargs):
        assert x is not None and x.ndim == 2

        results = []
        for head in self.heads:
            results.append(head(x=x, device=device))
        assert results != [] and [results[0].shape] * len(results) == [result.shape for result in results]

        if self.head_fusion is not None:
            assert self.head_fusion.get_num() == len(results) and [results[0].shape] * len(results) == [result.shape for result in results]
            result = self.head_fusion(x=results, w=self.w_head_fusion, device=device)
        else:
            assert len(results) == 1
            result = results[0]

        return result
