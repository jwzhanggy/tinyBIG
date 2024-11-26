# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################
# Chain based RPN Layer Module #
################################

"""
Chain Structural RPN based layers.

This module contains the chain structural rpn based layers, including
    graph_interdependence_layer

"""

import torch

from tinybig.module.base_layer import layer
from tinybig.head.chain_based_heads import chain_interdependence_head


class chain_interdependence_layer(layer):
    """
    A chain interdependence layer for capturing sequential dependencies in data.

    This layer integrates multiple chain interdependence heads to model sequential interdependencies.
    It supports features such as multi-hop connections, inverse or exponential approximations,
    parameter reconciliation, and various output processing functions.

    Attributes
    ----------
    m : int
        The input dimension of the layer.
    n : int
        The output dimension of the layer.
    chain_length : int
        The length of the chain for modeling interdependencies.
    channel_num : int
        The number of channels in each chain interdependence head.
    width : int
        The number of chain interdependence heads in the layer.
    name : str
        The name of the layer.
    bi_directional : bool
        Whether to include bi-directional dependencies in the chain.
    with_multihop : bool
        Whether to enable multi-hop dependencies.
    h : int
        The number of hops for multi-hop connections.
    accumulative : bool
        Whether to accumulate dependencies across hops.
    with_inverse_approx : bool
        Whether to use inverse approximation for interdependence.
    with_exponential_approx : bool
        Whether to use exponential approximation for interdependence.
    self_dependence : bool
        Whether to include self-dependencies in the chain.
    self_scaling : float
        The scaling factor for self-dependencies.
    with_dual_lphm : bool
        Whether to use dual LPHM reconciliation for parameters.
    with_lorr : bool
        Whether to use LORR reconciliation for parameters.
    r : int
        The rank for parameter reconciliation.
    enable_bias : bool
        Whether to enable bias in parameter reconciliation.
    with_residual : bool
        Whether to include a residual connection in the layer.
    with_batch_norm : bool
        Whether to apply batch normalization to the output.
    with_relu : bool
        Whether to apply ReLU activation to the output.
    with_dropout : bool
        Whether to apply dropout to the output.
    p : float
        Dropout probability.
    with_softmax : bool
        Whether to apply softmax activation to the output.
    parameters_init_method : str
        The initialization method for parameters.
    device : str
        The device to run the layer on ('cpu' or 'cuda').

    Methods
    -------
    __init__(...)
        Initializes the chain interdependence layer with specified parameters.
    forward(x, fusion_strategy='average', device='cpu', *args, **kwargs)
        Performs a forward pass through the layer.
    """
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
        """
        Initializes the chain interdependence layer.

        Parameters
        ----------
        m : int
            The input dimension of the layer.
        n : int
            The output dimension of the layer.
        chain_length : int
            The length of the chain for modeling interdependencies.
        channel_num : int, default=1
            The number of channels in each chain interdependence head.
        width : int, default=1
            The number of chain interdependence heads in the layer.
        name : str, default='chain_interdependence_layer'
            The name of the layer.
        bi_directional : bool, default=False
            Whether to include bi-directional dependencies in the chain.
        with_multihop : bool, default=False
            Whether to enable multi-hop dependencies.
        h : int, default=1
            The number of hops for multi-hop connections.
        accumulative : bool, default=False
            Whether to accumulate dependencies across hops.
        with_inverse_approx : bool, default=False
            Whether to use inverse approximation for interdependence.
        with_exponential_approx : bool, default=False
            Whether to use exponential approximation for interdependence.
        self_dependence : bool, default=True
            Whether to include self-dependencies in the chain.
        self_scaling : float, default=1.0
            The scaling factor for self-dependencies.
        with_dual_lphm : bool, default=False
            Whether to use dual LPHM reconciliation for parameters.
        with_lorr : bool, default=False
            Whether to use LORR reconciliation for parameters.
        r : int, default=3
            The rank for parameter reconciliation.
        enable_bias : bool, default=False
            Whether to enable bias in parameter reconciliation.
        with_residual : bool, default=False
            Whether to include a residual connection in the layer.
        with_batch_norm : bool, default=False
            Whether to apply batch normalization to the output.
        with_relu : bool, default=True
            Whether to apply ReLU activation to the output.
        with_dropout : bool, default=False
            Whether to apply dropout to the output.
        p : float, default=0.25
            Dropout probability.
        with_softmax : bool, default=True
            Whether to apply softmax activation to the output.
        parameters_init_method : str, default='xavier_normal'
            The initialization method for parameters.
        device : str, default='cpu'
            The device to run the layer on ('cpu' or 'cuda').

        Returns
        -------
        None
        """
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
        """
        Performs a forward pass through the chain interdependence layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape `(batch_size, m)`.
        fusion_strategy : str, default='average'
            The strategy for fusing outputs from multiple heads.
        device : str, default='cpu'
            The device to run the computation on ('cpu' or 'cuda').

        Returns
        -------
        torch.Tensor
            The output tensor with shape `(batch_size, n)` after applying the layer.
        """
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
