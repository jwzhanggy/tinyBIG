# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################
# RPN based Bayesian Network #
##############################

"""
RPN based machine learning models

This module contains the implementation of the RPN based machine learning models, including
    pgm
"""

import warnings

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import pgm_layer


class pgm(rpn):
    """
    A Probabilistic Graphical Model (PGM) implemented using the RPN framework.

    Parameters
    ----------
    dims : list[int] | tuple[int]
        A list or tuple of integers representing the dimensions of each layer in the PGM model.
        Must contain at least two dimensions.
    name : str, optional
        The name of the PGM model. Default is 'rpn_bayesian_network'.
    distribution : str, optional
        The type of distribution to use for data transformation. Supported options are:
        'normal', 'exponential', 'cauchy', 'gamma', 'laplace', and 'chi2'. Default is 'normal'.
    d : int, optional
        The degree for combinatorial expansion in the distribution. Default is 2.
    with_replacement : bool, optional
        Whether to allow replacement in combinatorial expansion. Default is False.
    enable_bias : bool, optional
        Whether to enable bias in the layers. Default is False.
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
    Warning
        If `dims` contains more than two dimensions, as regular Bayesian networks are typically shallow.

    Methods
    -------
    __init__(dims, name='rpn_bayesian_network', distribution='normal', ...)
        Initializes the PGM model and builds its layers.
    """
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_bayesian_network',
        distribution: str = 'normal',
        d: int = 2, with_replacement: bool = False,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device='cpu', *args, **kwargs
    ):
        """
        Initializes the Probabilistic Graphical Model (PGM).

        Parameters
        ----------
        dims : list[int] | tuple[int]
            A list or tuple of integers representing the dimensions of each layer in the PGM model.
            Must contain at least two dimensions.
        name : str, optional
            The name of the PGM model. Default is 'rpn_bayesian_network'.
        distribution : str, optional
            The type of distribution to use for data transformation. Supported options are:
            'normal', 'exponential', 'cauchy', 'gamma', 'laplace', and 'chi2'. Default is 'normal'.
        d : int, optional
            The degree for combinatorial expansion in the distribution. Default is 2.
        with_replacement : bool, optional
            Whether to allow replacement in combinatorial expansion. Default is False.
        enable_bias : bool, optional
            Whether to enable bias in the layers. Default is False.
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
        Warning
            If `dims` contains more than two dimensions, as regular Bayesian networks are typically shallow.
        """
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        if len(dims) > 2:
            warnings.warn("Regular SVMs should have only two layers...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                pgm_layer(
                    m=m, n=n,
                    enable_bias=enable_bias,
                    distribution=distribution,
                    d=d, with_replacement=with_replacement,
                    with_lorr=with_lorr, r=r,
                    with_residual=with_residual,
                    channel_num=channel_num,
                    width=width,
                    device=device, *args, **kwargs
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

