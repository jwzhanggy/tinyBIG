# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based RNN Model #
#######################

"""
RPN based RNN models

This module contains the implementation of the RPN based Transformer models, including
    transformer
"""

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer
from tinybig.layer.bilinear_layers import bilinear_interdependence_layer


class transformer(rpn):
    """
    Transformer model implemented as the RPN model.

    This class defines a transformer architecture with bilinear interdependence layers, perceptron layers,
    and fully connected layers. It supports various customization options for data transformation,
    parameter reconciliation, output processing, and interdependence functions.

    Parameters
    ----------
    dims : list[int] | tuple[int]
        A list or tuple of integers representing the dimensions of the bilinear interdependence layers.
        Must contain at least two dimensions.
    fc_dims : list[int] | tuple[int]
        A list or tuple of integers representing the dimensions of the fully connected layers.
        Must contain at least one dimension.
    batch_num : int, optional
        The batch size for bilinear interdependence layers. Default is None.
    name : str, optional
        The name of the transformer model. Default is 'rpn_transformer'.
    channel_num : int, optional
        The number of channels for each layer. Default is 1.
    width : int, optional
        The number of parallel heads in each layer. Default is 1.
    with_dual_lphm_interdependence : bool, optional
        If True, enables dual low-parametric high-order interdependence. Default is False.
    with_lorr_interdependence : bool, optional
        If True, enables low-rank parameterized interdependence. Default is True.
    r_interdependence : int, optional
        Rank parameter for low-rank interdependence. Default is 3.
    with_taylor : bool, optional
        If True, enables Taylor series-based data transformation. Default is False.
    d : int, optional
        Degree of Taylor series expansion. Default is 2.
    with_dual_lphm : bool, optional
        If True, enables dual low-parametric high-order parameter reconciliation. Default is False.
    with_lorr : bool, optional
        If True, enables low-rank parameterized reconciliation. Default is False.
    r : int, optional
        Rank parameter for low-rank reconciliation. Default is 3.
    enable_bias : bool, optional
        If True, enables bias in the layers. Default is True.
    with_residual : bool, optional
        If True, adds residual connections to the layers. Default is True.
    with_batch_norm : bool, optional
        If True, applies batch normalization after each layer. Default is True.
    with_relu : bool, optional
        If True, applies ReLU activation after each layer. Default is True.
    with_softmax : bool, optional
        If True, applies Softmax activation at the output layer. Default is True.
    with_dropout : bool, optional
        If True, applies dropout regularization. Default is True.
    p : float, optional
        Dropout probability. Default is 0.25.
    device : str, optional
        Device to perform computations ('cpu' or 'cuda'). Default is 'cpu'.
    *args : optional
        Additional positional arguments for the superclass.
    **kwargs : optional
        Additional keyword arguments for the superclass.

    Raises
    ------
    ValueError
        If `dims` contains fewer than two dimensions or `fc_dims` contains fewer than one dimension.

    Methods
    -------
    __init__(dims, fc_dims, batch_num, name, channel_num, width, ...)
        Initializes the transformer model with the specified parameters.
    """

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
        """
        Initialize the transformer model as the RPN model.

        Parameters
        ----------
        dims : list[int] | tuple[int]
            A list or tuple of integers representing the dimensions of the bilinear interdependence layers.
            Must contain at least two dimensions.
        fc_dims : list[int] | tuple[int]
            A list or tuple of integers representing the dimensions of the fully connected layers.
            Must contain at least one dimension.
        batch_num : int, optional
            The batch size for bilinear interdependence layers. Default is None.
        name : str, optional
            The name of the transformer model. Default is 'rpn_transformer'.
        channel_num : int, optional
            The number of channels for each layer. Default is 1.
        width : int, optional
            The number of parallel heads in each layer. Default is 1.
        with_dual_lphm_interdependence : bool, optional
            If True, enables dual low-parametric high-order interdependence. Default is False.
        with_lorr_interdependence : bool, optional
            If True, enables low-rank parameterized interdependence. Default is True.
        r_interdependence : int, optional
            Rank parameter for low-rank interdependence. Default is 3.
        with_taylor : bool, optional
            If True, enables Taylor series-based data transformation. Default is False.
        d : int, optional
            Degree of Taylor series expansion. Default is 2.
        with_dual_lphm : bool, optional
            If True, enables dual low-parametric high-order parameter reconciliation. Default is False.
        with_lorr : bool, optional
            If True, enables low-rank parameterized reconciliation. Default is False.
        r : int, optional
            Rank parameter for low-rank reconciliation. Default is 3.
        enable_bias : bool, optional
            If True, enables bias in the layers. Default is True.
        with_residual : bool, optional
            If True, adds residual connections to the layers. Default is True.
        with_batch_norm : bool, optional
            If True, applies batch normalization after each layer. Default is True.
        with_relu : bool, optional
            If True, applies ReLU activation after each layer. Default is True.
        with_softmax : bool, optional
            If True, applies Softmax activation at the output layer. Default is True.
        with_dropout : bool, optional
            If True, applies dropout regularization. Default is True.
        p : float, optional
            Dropout probability. Default is 0.25.
        device : str, optional
            Device to perform computations ('cpu' or 'cuda'). Default is 'cpu'.
        *args : optional
            Additional positional arguments for the superclass.
        **kwargs : optional
            Additional keyword arguments for the superclass.

        Raises
        ------
        ValueError
            If `dims` contains fewer than two dimensions or `fc_dims` contains fewer than one dimension.
        """
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


