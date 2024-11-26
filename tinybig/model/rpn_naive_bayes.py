# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# RPN based Naive Bayes Model #
###############################

"""
RPN based machine learning models

This module contains the implementation of the RPN based machine learning models, including
    naive_bayes
"""

import warnings

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import naive_bayes_layer


class naive_bayes(rpn):
    """
    A Naive Bayes model implemented using the RPN framework.

    Parameters
    ----------
    dims : list[int] | tuple[int]
        A list or tuple of integers representing the dimensions of each layer in the Naive Bayes model.
        Must contain at least two dimensions.
    name : str, optional
        The name of the Naive Bayes model. Default is 'rpn_naive_bayes'.
    distribution : str, optional
        The type of distribution to use for data transformation. Supported options are:
        'normal', 'exponential', 'cauchy', 'gamma', 'laplace', and 'chi2'. Default is 'normal'.
    enable_bias : bool, optional
        Whether to enable bias in the layers. Default is False.
    device : str, optional
        The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
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
    *args : optional
        Additional positional arguments for the `rpn` superclass.
    **kwargs : optional
        Additional keyword arguments for the `rpn` superclass.

    Raises
    ------
    ValueError
        If `dims` contains fewer than two dimensions.
    Warning
        If `dims` contains more than two dimensions, as regular Naive Bayes models are typically shallow.

    Methods
    -------
    __init__(dims, name='rpn_naive_bayes', distribution='normal', ...)
        Initializes the Naive Bayes model and builds its layers.
    """
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_naive_bayes',
        distribution: str = 'normal',
        enable_bias: bool = False,
        device: str = 'cpu',
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        *args, **kwargs
    ):
        """
        Initializes the Naive Bayes model.

        Parameters
        ----------
        dims : list[int] | tuple[int]
            A list or tuple of integers representing the dimensions of each layer in the Naive Bayes model.
            Must contain at least two dimensions.
        name : str, optional
            The name of the Naive Bayes model. Default is 'rpn_naive_bayes'.
        distribution : str, optional
            The type of distribution to use for data transformation. Supported options are:
            'normal', 'exponential', 'cauchy', 'gamma', 'laplace', and 'chi2'. Default is 'normal'.
        enable_bias : bool, optional
            Whether to enable bias in the layers. Default is False.
        device : str, optional
            The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
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
        *args : optional
            Additional positional arguments for the `rpn` superclass.
        **kwargs : optional
            Additional keyword arguments for the `rpn` superclass.

        Raises
        ------
        ValueError
            If `dims` contains fewer than two dimensions.
        Warning
            If `dims` contains more than two dimensions, as regular Naive Bayes models are typically shallow.
        """
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        if len(dims) > 2:
            warnings.warn("Regular SVMs should have only two layers...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                naive_bayes_layer(
                    m=m, n=n, device=device,
                    enable_bias=enable_bias,
                    distribution=distribution,
                    with_lorr=with_lorr,
                    with_residual=with_residual,
                    r=r,
                    channel_num=channel_num,
                    width=width,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

