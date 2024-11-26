# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based MLP Model #
#######################

"""
RPN based deep models

This module contains the implementation of the RPN based deep models, including
    kan
"""

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import kan_layer


class kan(rpn):
    """
    A model implementing the KAN architecture using RPN architecture.

    Parameters
    ----------
    dims : list[int] | tuple[int]
        A list or tuple of integers representing the dimensions of each layer in the KAN model.
        Must contain at least two dimensions.
    name : str, optional
        The name of the KAN model. Default is 'rpn_kan'.
    grid_range : tuple, optional
        The range of the grid for B-spline expansion. Default is (-1, 1).
    t : int, optional
        The number of grid points. Default is 5.
    d : int, optional
        The degree of the B-spline expansion. Default is 3.
    enable_bias : bool, optional
        Whether to enable bias in the layers. Default is False.
    with_lorr : bool, optional
        Whether to enable LoRR parameter reconciliation in the layers. Default is False.
    r : int, optional
        The rank parameter for the LoRR parameter reconciliation. Default is 3.
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
    __init__(dims, name='rpn_kan', grid_range=(-1, 1), t=5, d=3, ...)
        Initializes the KAN model and builds its layers.
    """
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_kan',
        grid_range=(-1, 1), t: int = 5, d: int = 3,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False, r: int = 3,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the KAN model.

        Parameters
        ----------
        dims : list[int] | tuple[int]
            A list or tuple of integers representing the dimensions of each layer in the KAN model.
            Must contain at least two dimensions.
        name : str, optional
            The name of the KAN model. Default is 'rpn_kan'.
        grid_range : tuple, optional
            The range of the grid for B-spline expansion. Default is (-1, 1).
        t : int, optional
            The number of grid points. Default is 5.
        d : int, optional
            The degree of the B-spline expansion. Default is 3.
        enable_bias : bool, optional
            Whether to enable bias in the layers. Default is False.
        with_lorr : bool, optional
            Whether to enable LoRR parameter reconciliation in the layers. Default is False.
        r : int, optional
            The rank parameter for the LoRR parameter reconciliation. Default is 3.
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
                kan_layer(
                    m=m, n=n,
                    grid_range=grid_range,
                    t=t, d=d,
                    enable_bias=enable_bias,
                    with_lorr=with_lorr, r=r,
                    device=device,
                    channel_num=channel_num,
                    width=width,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

