# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based RNN Model #
#######################
"""
RPN based RNN models

This module contains the implementation of the RPN based RNN models, including
    rnn
"""

import torch

import tinybig.layer
from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer
from tinybig.layer.chain_based_layers import chain_interdependence_layer


class rnn(rpn):
    """
    Recurrent Neural Network (RNN) model implemented as the RPN model.

    This class constructs an RNN model with a chain structure that supports interdependence functions and flexible
    data transformation. It allows for customized layer compositions and supports bidirectional processing,
    multi-hop interdependence, and various activation functions.

    Parameters
    ----------
    chain_length : int
        The length of the chain structure for the RNN layers.
    dims : list[int] | tuple[int]
        A list or tuple of integers representing the dimensions of each layer.
        Must contain at least two dimensions.
    name : str, optional
        The name of the RNN model. Default is 'rpn_rnn'.
    channel_num : int, optional
        The number of channels for each layer. Default is 1.
    width : int, optional
        The number of parallel heads in each layer. Default is 1.
    bi_directional : bool, optional
        If True, enables bidirectional processing in the chain structure. Default is False.
    with_multihop : bool, optional
        If True, enables multi-hop interdependence in the chain structure. Default is False.
    h : int, optional
        Number of hops for multi-hop interdependence. Default is 1.
    accumulative : bool, optional
        If True, accumulates multi-hop dependencies. Default is False.
    with_inverse_approx : bool, optional
        If True, enables inverse approximation for interdependence. Default is False.
    with_exponential_approx : bool, optional
        If True, enables exponential approximation for interdependence. Default is False.
    self_dependence : bool, optional
        If True, enables self-dependence in the chain structure. Default is True.
    self_scaling : float, optional
        Scaling factor for self-dependence. Default is 1.0.
    with_bspline : bool, optional
        If True, enables B-spline expansion for data transformation. Default is False.
    with_taylor : bool, optional
        If True, enables Taylor expansion for data transformation. Default is False.
    d : int, optional
        Degree of the expansion function (B-spline or Taylor). Default is 2.
    with_hybrid_expansion : bool, optional
        If True, enables hybrid data expansion. Default is False.
    with_dual_lphm : bool, optional
        If True, enables dual low-parametric hypermatrix reconciliation. Default is False.
    with_lorr : bool, optional
        If True, enables low-rank parameterized reconciliation. Default is False.
    r : int, optional
        Rank parameter for low-rank reconciliation. Default is 3.
    with_residual : bool, optional
        If True, adds residual connections to the layers. Default is False.
    with_dual_lphm_interdependence : bool, optional
        If True, enables dual low-parametric hypermatrix interdependence. Default is False.
    with_lorr_interdependence : bool, optional
        If True, enables low-rank interdependence. Default is False.
    r_interdependence : int, optional
        Rank for low-rank interdependence. Default is 3.
    enable_bias : bool, optional
        If True, enables bias in the layers. Default is False.
    with_batch_norm : bool, optional
        If True, applies batch normalization to the layers. Default is False.
    with_relu : bool, optional
        If True, applies ReLU activation to the layers. Default is True.
    with_softmax : bool, optional
        If True, applies Softmax activation to the output layer. Default is True.
    with_dropout : bool, optional
        If True, applies dropout to the layers. Default is False.
    p : float, optional
        Dropout probability. Default is 0.25.
    parameters_init_method : str, optional
        Initialization method for the parameters. Default is 'xavier_normal'.
    device : str, optional
        Device to perform computations ('cpu' or 'cuda'). Default is 'cpu'.
    *args : optional
        Additional positional arguments for the superclass.
    **kwargs : optional
        Additional keyword arguments for the superclass.

    Raises
    ------
    ValueError
        If `dims` is empty or contains fewer than two dimensions.

    Methods
    -------
    __init__(...)
        Initializes the RNN model with the specified parameters.
    forward(x: torch.Tensor, device='cpu', *args, **kwargs)
        Performs a forward pass through the RNN model.
    """
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
        # data expansion function
        with_bspline: bool = False,
        with_taylor: bool = False, d: int = 2,
        with_hybrid_expansion: bool = False,
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
        """
        Initialize the RNN model as a RPN model.

        Parameters
        ----------
        chain_length : int
            The length of the chain structure for the RNN layers.
        dims : list[int] | tuple[int]
            A list or tuple of integers representing the dimensions of each layer.
            Must contain at least two dimensions.
        name : str, optional
            The name of the RNN model. Default is 'rpn_rnn'.
        channel_num : int, optional
            The number of channels for each layer. Default is 1.
        width : int, optional
            The number of parallel heads in each layer. Default is 1.
        bi_directional : bool, optional
            If True, enables bidirectional processing in the chain structure. Default is False.
        with_multihop : bool, optional
            If True, enables multi-hop interdependence in the chain structure. Default is False.
        h : int, optional
            Number of hops for multi-hop interdependence. Default is 1.
        accumulative : bool, optional
            If True, accumulates multi-hop dependencies. Default is False.
        with_inverse_approx : bool, optional
            If True, enables inverse approximation for interdependence. Default is False.
        with_exponential_approx : bool, optional
            If True, enables exponential approximation for interdependence. Default is False.
        self_dependence : bool, optional
            If True, enables self-dependence in the chain structure. Default is True.
        self_scaling : float, optional
            Scaling factor for self-dependence. Default is 1.0.
        with_bspline : bool, optional
            If True, enables B-spline expansion for data transformation. Default is False.
        with_taylor : bool, optional
            If True, enables Taylor expansion for data transformation. Default is False.
        d : int, optional
            Degree of the expansion function (B-spline or Taylor). Default is 2.
        with_hybrid_expansion : bool, optional
            If True, enables hybrid data expansion. Default is False.
        with_dual_lphm : bool, optional
            If True, enables dual low-parametric hypermatrix reconciliation. Default is False.
        with_lorr : bool, optional
            If True, enables low-rank parameterized reconciliation. Default is False.
        r : int, optional
            Rank parameter for low-rank reconciliation. Default is 3.
        with_residual : bool, optional
            If True, adds residual connections to the layers. Default is False.
        with_dual_lphm_interdependence : bool, optional
            If True, enables dual low-parametric hypermatrix interdependence. Default is False.
        with_lorr_interdependence : bool, optional
            If True, enables low-rank interdependence. Default is False.
        r_interdependence : int, optional
            Rank for low-rank interdependence. Default is 3.
        enable_bias : bool, optional
            If True, enables bias in the layers. Default is False.
        with_batch_norm : bool, optional
            If True, applies batch normalization to the layers. Default is False.
        with_relu : bool, optional
            If True, applies ReLU activation to the layers. Default is True.
        with_softmax : bool, optional
            If True, applies Softmax activation to the output layer. Default is True.
        with_dropout : bool, optional
            If True, applies dropout to the layers. Default is False.
        p : float, optional
            Dropout probability. Default is 0.25.
        parameters_init_method : str, optional
            Initialization method for the parameters. Default is 'xavier_normal'.
        device : str, optional
            Device to perform computations ('cpu' or 'cuda'). Default is 'cpu'.
        *args : optional
            Additional positional arguments for the superclass.
        **kwargs : optional
            Additional keyword arguments for the superclass.

        Raises
        ------
        ValueError
            If `dims` is empty or contains fewer than two dimensions.
        """
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
                    with_bspline=with_bspline,
                    with_taylor=with_taylor, d=d,
                    with_hybrid_expansion=with_hybrid_expansion,
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
                with_bspline=with_bspline,
                with_taylor=with_taylor, d=d,
                with_hybrid_expansion=with_hybrid_expansion,
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
        """
        Perform a forward pass through the RNN model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape `(batch_size, input_dim)`.
        device : str, optional
            The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
        *args : optional
            Additional positional arguments.
        **kwargs : optional
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the RNN model.
        """
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

