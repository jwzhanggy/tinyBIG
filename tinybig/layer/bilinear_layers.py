# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

############################
# Transformer Layer Module #
############################

"""
Bilinear RPN based layers.

This module contains the bilinear rpn based layers, including
    bilinear_interdependence_layer
"""

from tinybig.module.base_layer import layer
from tinybig.head.bilinear_heads import bilinear_interdependence_head
from tinybig.fusion.parameterized_concatenation_fusion import parameterized_concatenation_fusion


class bilinear_interdependence_layer(layer):
    """
    A bilinear interdependence layer for processing data with interdependencies.

    This layer incorporates bilinear interdependence heads with optional features such as Taylor expansions,
    parameter reconciliation, and various output processing functions. It supports channel fusion using
    parameterized concatenation.

    Attributes
    ----------
    m : int
        The input dimension of the layer.
    n : int
        The output dimension of the layer.
    name : str
        The name of the layer.
    batch_num : int
        The number of batches for instance interdependence.
    channel_num : int
        The number of channels in the layer.
    width : int
        The number of bilinear interdependence heads in the layer.
    with_dual_lphm_interdependence : bool
        Whether to use dual LPHM interdependence.
    with_lorr_interdependence : bool
        Whether to use LORR interdependence.
    r_interdependence : int
        The rank for bilinear interdependence.
    with_taylor : bool
        Whether to use Taylor expansion for data transformation.
    d : int
        The degree of the Taylor expansion.
    with_dual_lphm : bool
        Whether to use dual LPHM reconciliation for parameters.
    with_lorr : bool
        Whether to use LORR reconciliation for parameters.
    r : int
        The rank for parameter reconciliation.
    enable_bias : bool
        Whether to enable bias in parameter reconciliation.
    with_residual : bool
        Whether to include a residual connection.
    with_batch_norm : bool
        Whether to apply batch normalization to the output.
    with_relu : bool
        Whether to apply ReLU activation to the output.
    with_softmax : bool
        Whether to apply softmax activation to the output.
    with_dropout : bool
        Whether to apply dropout to the output.
    p : float
        Dropout probability.
    parameters_init_method : str
        The initialization method for parameters.
    device : str
        The device to run the layer on ('cpu' or 'cuda').
    head_fusion : parameterized_concatenation_fusion
        The fusion method for combining outputs from multiple heads.
    """
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
        """
        Initialize a bilinear interdependence layer.

        Parameters
        ----------
        m : int
            The input dimension of the layer.
        n : int
            The output dimension of the layer.
        name : str, default='attention_layer'
            The name of the layer.
        batch_num : int, optional
            The number of batches for instance interdependence.
        channel_num : int, default=1
            The number of channels in the layer.
        width : int, default=1
            The number of bilinear interdependence heads in the layer.
        with_dual_lphm_interdependence : bool, default=False
            Whether to use dual LPHM interdependence.
        with_lorr_interdependence : bool, default=False
            Whether to use LORR interdependence.
        r_interdependence : int, default=3
            The rank for bilinear interdependence.
        with_taylor : bool, default=False
            Whether to use Taylor expansion for data transformation.
        d : int, default=2
            The degree of the Taylor expansion.
        with_dual_lphm : bool, default=False
            Whether to use dual LPHM reconciliation for parameters.
        with_lorr : bool, default=False
            Whether to use LORR reconciliation for parameters.
        r : int, default=3
            The rank for parameter reconciliation.
        enable_bias : bool, default=False
            Whether to enable bias in parameter reconciliation.
        with_residual : bool, default=False
            Whether to include a residual connection.
        with_batch_norm : bool, default=False
            Whether to apply batch normalization to the output.
        with_relu : bool, default=True
            Whether to apply ReLU activation to the output.
        with_softmax : bool, default=True
            Whether to apply softmax activation to the output.
        with_dropout : bool, default=False
            Whether to apply dropout to the output.
        p : float, default=0.25
            Dropout probability.
        parameters_init_method : str, default='xavier_normal'
            The initialization method for parameters.
        device : str, default='cpu'
            The device to run the layer on ('cpu' or 'cuda').

        Returns
        -------
        None
        """
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
