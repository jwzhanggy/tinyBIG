# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based CNN Model #
#######################

"""
RPN based CNN models

This module contains the implementation of the RPN based CNN models, including
    cnn
    resnet
"""

from tinybig.model.rpn import rpn
from tinybig.layer.basic_layers import perceptron_layer
from tinybig.layer.grid_based_layers import grid_interdependence_layer, grid_compression_layer
from tinybig.util import parameter_scheduler


class cnn(rpn):
    """
    A Convolutional Neural Network (CNN) built on the RPN framework.

    This CNN architecture consists of:

    - Grid-based interdependence layers for convolution.
    - Optional pooling layers for feature map compression.
    - Fully connected perceptron layers for classification or regression.

    The architecture supports various configurations, including patch-based structures,
    pooling strategies, and advanced processing features like batch normalization,
    dropout, and activation functions.

    Attributes
    ----------
    h : int
        Height of the input grid.
    w : int
        Width of the input grid.
    channel_nums : list[int] | tuple[int]
        Sequence of channel numbers for each convolutional layer.
    fc_dims : list[int] | tuple[int]
        Dimensions of the fully connected layers.
    d : int
        Depth of the input grid. Default is 1 (2D input).
    fc_channel_num : int
        Number of channels for the fully connected layers. Default is 1.
    width : int
        Number of parallel heads for each grid interdependence layer. Default is 1.
    pooling_metric : str
        Pooling metric used in pooling layers (e.g., 'batch_max'). Default is 'batch_max'.
    pooling_layer_gaps : int
        Number of layers before adding a pooling layer. Default is 2.
    patch_size_half_after_pooling : bool
        If True, reduces patch size by half after pooling. Default is False.
    patch_shape : str
        Shape of the patches ('cuboid', 'cylinder', or 'sphere'). Default is 'cuboid'.
    with_batch_norm : bool
        If True, applies batch normalization. Default is True.
    with_relu : bool
        If True, applies ReLU activation. Default is True.
    with_softmax : bool
        If True, applies softmax activation to the final output. Default is False.
    with_residual : bool
        If True, includes residual connections. Default is False.
    with_dropout : bool
        If True, applies dropout. Default is True.
    device : str
        Device for computation ('cpu' or 'cuda'). Default is 'cpu'.

    Methods
    -------
    __init__(...)
        Initializes the CNN with grid-based convolutional layers, pooling layers, and fully connected layers.
    """
    def __init__(
        self,
        h: int, w: int,
        channel_nums: list[int] | tuple[int],
        fc_dims: list[int] | tuple[int],
        d: int = 1,
        fc_channel_num: int = 1,
        width: int = 1,
        pooling_metric: str = 'batch_max',
        pooling_layer_gaps: int = 2,
        patch_size_half_after_pooling: bool = False,
        name: str = 'rpn_cnn',
        # patch structure parameters for interdependence
        patch_shape: str = 'cuboid',
        p_h: int = None, p_h_prime: int = None,
        p_w: int = None, p_w_prime: int = None,
        p_d: int = 0, p_d_prime: int = None,
        p_r: int = None,
        cd_h: int = None, cd_w: int = None, cd_d: int = 1,
        packing_strategy: str = 'densest_packing',
        # patch structure parameters for compression
        pooling_patch_shape: str = None,
        pooling_p_h: int = None, pooling_p_h_prime: int = None,
        pooling_p_w: int = None, pooling_p_w_prime: int = None,
        pooling_p_d: int = None, pooling_p_d_prime: int = None,
        pooling_p_r: int = None,
        pooling_cd_h: int = None, pooling_cd_w: int = None, pooling_cd_d: int = None,
        pooling_packing_strategy: str = None,
        # output processing function parameters
        with_batch_norm: bool = True,
        with_relu: bool = True,
        with_softmax: bool = False,
        with_residual: bool = False,
        with_dropout: bool = True, p_pooling: float = 0.25, p_fc: float = 0.5,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = True,
        # parameter reconciliation function parameters
        with_perceptron_residual: bool = None,
        with_perceptron_dual_lphm: bool = None,
        with_perceptron_lorr: bool = None, perceptron_r: int = None,
        enable_perceptron_bias: bool = None,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the CNN (Convolutional Neural Network) model.

        This method constructs the CNN architecture with configurable convolutional layers, optional pooling layers,
        and fully connected perceptron layers. Various processing features like batch normalization, dropout,
        and residual connections can also be enabled.

        Parameters
        ----------
        h : int
            Height of the input grid.
        w : int
            Width of the input grid.
        channel_nums : list[int] | tuple[int]
            Sequence of channel numbers for each convolutional layer.
        fc_dims : list[int] | tuple[int]
            Dimensions of the fully connected layers.
        d : int, optional
            Depth of the input grid. Default is 1 (2D input).
        fc_channel_num : int, optional
            Number of channels for the fully connected layers. Default is 1.
        width : int, optional
            Number of parallel heads for each grid interdependence layer. Default is 1.
        pooling_metric : str, optional
            Pooling metric used in pooling layers (e.g., 'batch_max'). Default is 'batch_max'.
        pooling_layer_gaps : int, optional
            Number of layers before adding a pooling layer. Default is 2.
        patch_size_half_after_pooling : bool, optional
            If True, reduces patch size by half after pooling. Default is False.
        name : str, optional
            Name of the CNN model. Default is 'rpn_cnn'.
        patch_shape : str, optional
            Shape of the patches ('cuboid', 'cylinder', or 'sphere'). Default is 'cuboid'.
        p_h, p_h_prime : int, optional
            Height and height prime of the patches. Default is None.
        p_w, p_w_prime : int, optional
            Width and width prime of the patches. Default is None.
        p_d, p_d_prime : int, optional
            Depth and depth prime of the patches. Default is 0 and None, respectively.
        p_r : int, optional
            Radius of spherical or cylindrical patches. Default is None.
        cd_h, cd_w, cd_d : int, optional
            Compression dimensions for height, width, and depth, respectively. Default is 1.
        packing_strategy : str, optional
            Strategy for patch packing. Default is 'densest_packing'.
        pooling_patch_shape : str, optional
            Shape of pooling patches. Default is None (same as `patch_shape`).
        pooling_p_h, pooling_p_h_prime : int, optional
            Height and height prime of pooling patches. Default is None.
        pooling_p_w, pooling_p_w_prime : int, optional
            Width and width prime of pooling patches. Default is None.
        pooling_p_d, pooling_p_d_prime : int, optional
            Depth and depth prime of pooling patches. Default is None.
        pooling_p_r : int, optional
            Radius of pooling patches. Default is None.
        pooling_cd_h, pooling_cd_w, pooling_cd_d : int, optional
            Compression dimensions for pooling patches. Default is None.
        pooling_packing_strategy : str, optional
            Packing strategy for pooling patches. Default is None (same as `packing_strategy`).
        with_batch_norm : bool, optional
            If True, applies batch normalization. Default is True.
        with_relu : bool, optional
            If True, applies ReLU activation. Default is True.
        with_softmax : bool, optional
            If True, applies softmax activation to the final output. Default is False.
        with_residual : bool, optional
            If True, includes residual connections. Default is False.
        with_dropout : bool, optional
            If True, applies dropout. Default is True.
        p_pooling : float, optional
            Dropout probability for pooling layers. Default is 0.25.
        p_fc : float, optional
            Dropout probability for fully connected layers. Default is 0.5.
        with_dual_lphm : bool, optional
            If True, enables dual low-parametric high-order interdependence. Default is False.
        with_lorr : bool, optional
            If True, enables low-rank parameterized interdependence. Default is False.
        r : int, optional
            Rank parameter for low-rank interdependence. Default is 3.
        enable_bias : bool, optional
            If True, includes bias in the layers. Default is True.
        with_perceptron_residual : bool, optional
            If True, includes residual connections in perceptron layers. Default is None (inherits from `with_residual`).
        with_perceptron_dual_lphm : bool, optional
            If True, enables dual low-parametric high-order interdependence in perceptron layers. Default is None.
        with_perceptron_lorr : bool, optional
            If True, enables low-rank parameterized interdependence in perceptron layers. Default is None.
        perceptron_r : int, optional
            Rank parameter for perceptron layers. Default is None.
        enable_perceptron_bias : bool, optional
            If True, includes bias in perceptron layers. Default is None.
        device : str, optional
            Device for computation ('cpu' or 'cuda'). Default is 'cpu'.
        *args : optional
            Additional positional arguments for layers.
        **kwargs : optional
            Additional keyword arguments for layers.

        Raises
        ------
        ValueError
            If input dimensions are invalid or improperly configured.
        """
        print('############# rpn-cnn model architecture ############')

        if pooling_patch_shape is None: pooling_patch_shape = patch_shape
        if pooling_p_h is None: pooling_p_h = p_h
        if pooling_p_h_prime is None: pooling_p_h_prime = p_h_prime
        if pooling_p_w is None: pooling_p_w = p_w
        if pooling_p_w_prime is None: pooling_p_w_prime = p_w_prime
        if pooling_p_d is None: pooling_p_d = p_d
        if pooling_p_d_prime is None: pooling_p_d_prime = p_d_prime
        if pooling_p_r is None: pooling_p_r = p_r
        if pooling_cd_h is None: pooling_cd_h = cd_h
        if pooling_cd_w is None: pooling_cd_w = cd_w
        if pooling_cd_d is None: pooling_cd_d = cd_d
        if pooling_packing_strategy is None: pooling_packing_strategy = packing_strategy

        if with_perceptron_residual is None: with_perceptron_residual = with_residual
        if with_perceptron_dual_lphm is None: with_perceptron_dual_lphm = with_dual_lphm
        if with_perceptron_lorr is None: with_perceptron_lorr = with_lorr
        if perceptron_r is None: perceptron_r = r
        if enable_perceptron_bias is None: enable_perceptron_bias = enable_bias

        layers = []
        for in_channel, out_channel in zip(channel_nums, channel_nums[1:]):
            print('conv in', h, w, d, in_channel)
            layer = grid_interdependence_layer(
                h=h, w=w, d=d,
                in_channel=in_channel, out_channel=out_channel,
                width=width,
                patch_shape=patch_shape,
                p_h=p_h, p_h_prime=p_h_prime,
                p_w=p_w, p_w_prime=p_w_prime,
                p_d=p_d, p_d_prime=p_d_prime,
                p_r=p_r,
                cd_h=cd_h, cd_w=cd_w, cd_d=cd_d,
                packing_strategy=packing_strategy,
                with_batch_norm=with_batch_norm,
                with_relu=with_relu,
                with_residual=with_residual,
                enable_bias=enable_bias,
                with_dual_lphm=with_dual_lphm,
                with_lorr=with_lorr, r=r,
                device=device, *args, **kwargs
            )
            h, w, d = layer.get_output_grid_shape()
            print('conv out', h, w, d, out_channel)
            layers.append(layer)

            # adding a pooling layer for a certain layer gaps
            if len(layers) % (pooling_layer_gaps+1) == pooling_layer_gaps:
                print('pooling in', h, w, d, out_channel)
                layer = grid_compression_layer(
                    h=h, w=w, d=d,
                    channel_num=out_channel,
                    pooling_metric=pooling_metric,
                    patch_shape=pooling_patch_shape,
                    p_h=pooling_p_h, p_h_prime=pooling_p_h_prime,
                    p_w=pooling_p_w, p_w_prime=pooling_p_w_prime,
                    p_d=pooling_p_d, p_d_prime=pooling_p_d_prime,
                    p_r=pooling_p_r,
                    cd_h=pooling_cd_h, cd_w=pooling_cd_w, cd_d=pooling_cd_d,
                    packing_strategy=pooling_packing_strategy,
                    with_dropout=with_dropout, p=p_pooling,
                    device=device, *args, **kwargs
                )
                h, w, d = layer.get_output_grid_shape()
                print('pooling out', h, w, d, out_channel)
                layers.append(layer)

                if patch_size_half_after_pooling:
                    print('halving patch size')
                    p_h, p_h_prime, p_w, p_w_prime, p_d, p_d_prime, p_r = parameter_scheduler(strategy='half', parameter_list=[p_h, p_h_prime, p_w, p_w_prime, p_d, p_d_prime, p_r])
                    pooling_p_h, pooling_p_h_prime, pooling_p_w, pooling_p_w_prime, pooling_p_d, pooling_p_d_prime, pooling_p_r = parameter_scheduler(strategy='half', parameter_list=[pooling_p_h, pooling_p_h_prime, pooling_p_w, pooling_p_w_prime, pooling_p_d, pooling_p_d_prime, pooling_p_r])

        # perceptron layers
        assert len(layers) >= 1
        m = layers[-1].get_n()
        dims = [m] + fc_dims
        for m, n in zip(dims, dims[1:]):
            print('fc in', m)
            layers.append(
                perceptron_layer(
                    m=m, n=n,
                    enable_bias=enable_perceptron_bias,
                    with_dual_lphm=with_perceptron_dual_lphm,
                    with_lorr=with_perceptron_lorr, r=perceptron_r,
                    with_residual=with_perceptron_residual,
                    channel_num=fc_channel_num,
                    width=width,
                    with_batch_norm=with_batch_norm and n != dims[-1],
                    with_relu=with_relu and n != dims[-1],
                    with_dropout=with_dropout and n != dims[-1], p=p_fc,
                    with_softmax=with_softmax and m == dims[-2] and n == dims[-1],
                    device=device, *args, **kwargs
                )
            )
            print('fc out', n)
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)


class resnet(cnn):
    """
    A Residual Neural Network (ResNet) built on the CNN (Convolutional Neural Network) architecture.

    This ResNet model extends the CNN by including residual connections, enabling better optimization
    for deeper networks and preventing vanishing gradient issues.

    Attributes
    ----------
    with_residual : bool
        If True, includes residual connections in the layers. Default is True.

    Inherited Attributes
    ---------------------
    Inherits all attributes from the `cnn` class, including `h`, `w`, `channel_nums`,
    `fc_dims`, and processing features like batch normalization and dropout.

    Methods
    -------
    __init__(...)
        Initializes the ResNet model, enabling residual connections in the CNN architecture.
    """
    def __init__(self, with_residual: bool = True, *args, **kwargs):
        """
        Initialize the ResNet (Residual Network) model.

        This model extends the CNN architecture by enabling residual connections.

        Parameters
        ----------
        with_residual : bool, optional
            If True, includes residual connections in the layers. Default is True.
        *args, **kwargs
            Additional arguments passed to the parent CNN class.
        """
        super().__init__(with_residual=True, *args, **kwargs)

