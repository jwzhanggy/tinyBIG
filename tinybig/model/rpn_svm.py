# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based SVM Model #
#######################

"""
RPN based machine learning models

This module contains the implementation of the RPN based machine learning models, including
    svm
"""

import warnings

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import svm_layer


class svm(rpn):
    """
    Support Vector Machine (SVM) model implemented as the RPN model.

    This class defines an SVM model with customizable kernel functions, parameter reconciliation,
    and additional processing capabilities such as residual connections and multi-channel configurations.

    Parameters
    ----------
    dims : list[int] | tuple[int]
        A list or tuple of integers representing the dimensions of each layer.
        Must contain at least two dimensions.
    name : str, optional
        The name of the SVM model. Default is 'rpn_svm'.
    kernel : str, optional
        The kernel type to use. Options include 'linear', 'gaussian_rbf', and 'inverse_quadratic_rbf'.
        Default is 'linear'.
    base_range : tuple, optional
        The base range for kernel expansion. Default is (-1, 1).
    num_interval : int, optional
        The number of intervals for kernel expansion. Default is 10.
    epsilon : float, optional
        Parameter for kernel approximation. Default is 1.0.
    enable_bias : bool, optional
        If True, enables bias in the layers. Default is False.
    with_lorr : bool, optional
        If True, enables low-rank parameterized reconciliation. Default is False.
    r : int, optional
        Rank parameter for low-rank reconciliation. Default is 3.
    with_residual : bool, optional
        If True, adds residual connections to the layers. Default is False.
    channel_num : int, optional
        The number of channels for each layer. Default is 1.
    width : int, optional
        The number of parallel heads in each layer. Default is 1.
    device : str, optional
        Device to perform computations ('cpu' or 'cuda'). Default is 'cpu'.
    *args : optional
        Additional positional arguments for the superclass.
    **kwargs : optional
        Additional keyword arguments for the superclass.

    Raises
    ------
    ValueError
        If `dims` contains fewer than two dimensions.
    Warning
        If more than two layers are defined, as SVMs typically use only two layers.

    Methods
    -------
    __init__(dims, name, kernel, base_range, num_interval, epsilon, enable_bias, ...)
        Initializes the SVM model with the specified parameters.
    """
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_svm',
        kernel: str = 'linear',
        base_range: tuple = (-1, 1),
        num_interval: int = 10,
        epsilon: float = 1.0,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize the SVM model as a recursive parameterized network (RPN).

        Parameters
        ----------
        dims : list[int] | tuple[int]
            A list or tuple of integers representing the dimensions of each layer.
            Must contain at least two dimensions.
        name : str, optional
            The name of the SVM model. Default is 'rpn_svm'.
        kernel : str, optional
            The kernel type to use. Options include 'linear', 'gaussian_rbf', and 'inverse_quadratic_rbf'.
            Default is 'linear'.
        base_range : tuple, optional
            The base range for kernel expansion. Default is (-1, 1).
        num_interval : int, optional
            The number of intervals for kernel expansion. Default is 10.
        epsilon : float, optional
            Parameter for kernel approximation. Default is 1.0.
        enable_bias : bool, optional
            If True, enables bias in the layers. Default is False.
        with_lorr : bool, optional
            If True, enables low-rank parameterized reconciliation. Default is False.
        r : int, optional
            Rank parameter for low-rank reconciliation. Default is 3.
        with_residual : bool, optional
            If True, adds residual connections to the layers. Default is False.
        channel_num : int, optional
            The number of channels for each layer. Default is 1.
        width : int, optional
            The number of parallel heads in each layer. Default is 1.
        device : str, optional
            Device to perform computations ('cpu' or 'cuda'). Default is 'cpu'.
        *args : optional
            Additional positional arguments for the superclass.
        **kwargs : optional
            Additional keyword arguments for the superclass.

        Raises
        ------
        ValueError
            If `dims` contains fewer than two dimensions.
        Warning
            If more than two layers are defined, as SVMs typically use only two layers.
        """
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        if len(dims) > 2:
            warnings.warn("Regular SVMs should have only two layers...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                svm_layer(
                    m=m, n=n,
                    enable_bias=enable_bias,
                    kernel=kernel,
                    base_range=base_range,
                    num_interval=num_interval,
                    epsilon=epsilon,
                    with_lorr=with_lorr, r=r,
                    with_residual=with_residual,
                    device=device,
                    channel_num=channel_num,
                    width=width,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

