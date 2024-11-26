# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based MLP Model #
#######################

"""
RPN based deep models

This module contains the implementation of the RPN based deep models, including
    mlp
"""

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer


class mlp(rpn):
    """
    A Multi-Layer Perceptron (MLP) implemented using the RPN framework.

    Parameters
    ----------
    dims : list[int] | tuple[int]
        A list or tuple of integers representing the dimensions of each layer in the MLP.
        Must contain at least two dimensions.
    name : str, optional
        The name of the MLP model. Default is 'rpn_mlp'.
    enable_bias : bool, optional
        Whether to enable bias in the layers. Default is False.
    with_taylor : bool, optional
        Whether to use Taylor expansion for data transformation. Default is False.
    d : int, optional
        The degree of the Taylor expansion if `with_taylor` is True. Default is 2.
    with_lorr : bool, optional
        Whether to use LoRR (Low-Rank Representation) for parameter reconciliation. Default is False.
    r : int, optional
        The rank for the LoRR parameter reconciliation. Default is 3.
    with_residual : bool, optional
        Whether to enable residual connections in the layers. Default is False.
    channel_num : int, optional
        The number of channels in each layer. Default is 1.
    width : int, optional
        The number of parallel heads in each layer. Default is 1.
    device : str, optional
        The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
    *args : optional
        Additional positional arguments for the `rpn` superclass.
    **kwargs : optional
        Additional keyword arguments for the `rpn` superclass.

    Raises
    ------
    ValueError
        If `dims` contains fewer than two dimensions.

    Methods
    -------
    __init__(dims, name='rpn_mlp', enable_bias=False, ...)
        Initializes the MLP model and builds its layers.
    """
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_mlp',
        enable_bias: bool = False,
        # optional parameters
        with_taylor: bool = False,
        d: int = 2,
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the MLP model.

        Parameters
        ----------
        dims : list[int] | tuple[int]
            A list or tuple of integers representing the dimensions of each layer in the MLP.
            Must contain at least two dimensions.
        name : str, optional
            The name of the MLP model. Default is 'rpn_mlp'.
        enable_bias : bool, optional
            Whether to enable bias in the layers. Default is False.
        with_taylor : bool, optional
            Whether to use Taylor expansion for data transformation. Default is False.
        d : int, optional
            The degree of the Taylor expansion if `with_taylor` is True. Default is 2.
        with_lorr : bool, optional
            Whether to use LoRR (Low-Rank Representation) for parameter reconciliation. Default is False.
        r : int, optional
            The rank for the LoRR parameter reconciliation. Default is 3.
        with_residual : bool, optional
            Whether to enable residual connections in the layers. Default is False.
        channel_num : int, optional
            The number of channels in each layer. Default is 1.
        width : int, optional
            The number of parallel heads in each layer. Default is 1.
        device : str, optional
            The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
        *args : optional
            Additional positional arguments for the `rpn` superclass.
        **kwargs : optional
            Additional keyword arguments for the `rpn` superclass.

        Raises
        ------
        ValueError
            If `dims` contains fewer than two dimensions.
        """
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                perceptron_layer(
                    m=m, n=n, device=device,
                    enable_bias=enable_bias,
                    with_taylor=with_taylor,
                    with_lorr=with_lorr,
                    with_residual=with_residual,
                    d=d, r=r,
                    channel_num=channel_num,
                    width=width,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

