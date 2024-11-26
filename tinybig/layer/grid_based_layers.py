# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Grid based RPN Layer Module #
###############################
"""
Grid Structural RPN based layers.

This module contains the grid structural rpn based layers, including
    grid_interdependence_layer

"""

from tinybig.module.base_layer import layer
from tinybig.head.grid_based_heads import grid_interdependence_head, grid_compression_head
from tinybig.fusion.concatenation_fusion import concatenation_fusion
from tinybig.fusion.metric_fusion import mean_fusion


class grid_interdependence_layer(layer):
    """
    A layer for modeling grid-based interdependence with configurable patch structures and packing strategies.

    This layer supports operations over a grid input, where each head applies a grid-based interdependence function
    and aggregates results using optional fusion mechanisms.

    Parameters
    ----------
    h : int
        The height of the input grid.
    w : int
        The width of the input grid.
    in_channel : int
        The number of input channels.
    out_channel : int
        The number of output channels.
    d : int, optional
        The depth of the input grid. Default is 1 (2D grid).
    width : int, optional
        The number of grid interdependence heads. Default is 1.
    name : str, optional
        The name of the layer. Default is 'grid_interdependence_layer'.
    patch_shape : str, optional
        The shape of the patch ('cuboid', 'cylinder', or 'sphere'). Default is 'cuboid'.
    p_h : int, optional
        The height of the patch.
    p_h_prime : int, optional
        The height of the inner patch (for hierarchical packing).
    p_w : int, optional
        The width of the patch.
    p_w_prime : int, optional
        The width of the inner patch (for hierarchical packing).
    p_d : int, optional
        The depth of the patch (only for 3D grids).
    p_d_prime : int, optional
        The depth of the inner patch (for hierarchical packing).
    p_r : int, optional
        The radius of the patch (for spherical or cylindrical patches).
    cd_h : int, optional
        The compression degree along the height dimension.
    cd_w : int, optional
        The compression degree along the width dimension.
    cd_d : int, optional
        The compression degree along the depth dimension. Default is 1.
    packing_strategy : str, optional
        The packing strategy ('densest_packing' or other supported modes). Default is 'densest_packing'.
    with_batch_norm : bool, optional
        If True, applies batch normalization. Default is True.
    with_relu : bool, optional
        If True, applies ReLU activation. Default is True.
    with_residual : bool, optional
        If True, includes residual connections. Default is False.
    enable_bias : bool, optional
        If True, enables bias terms in parameter reconciliation. Default is False.
    with_dual_lphm : bool, optional
        If True, uses dual LPHM reconciliation for parameter fabrication. Default is False.
    with_lorr : bool, optional
        If True, uses LORR reconciliation for parameter fabrication. Default is False.
    r : int, optional
        The rank for parameter reconciliation. Default is 3.
    parameters_init_method : str, optional
        The initialization method for parameters. Default is 'xavier_normal'.
    device : str, optional
        The device for computations ('cpu' or 'cuda'). Default is 'cpu'.
    *args, **kwargs
        Additional arguments passed to the parent class.

    Methods
    -------
    get_output_grid_shape():
        Returns the shape of the output grid after packing.

    Raises
    ------
    ValueError
        If invalid patch shape, dimensions, or configuration is provided.
    """
    def __init__(
        self,
        h: int, w: int, in_channel: int, out_channel: int,
        d: int = 1,
        width: int = 1,
        name: str = 'grid_interdependence_layer',
        patch_shape: str = 'cuboid',
        p_h: int = None, p_h_prime: int = None,
        p_w: int = None, p_w_prime: int = None,
        p_d: int = 0, p_d_prime: int = None,
        p_r: int = None,
        cd_h: int = None, cd_w: int = None, cd_d: int = 1,
        packing_strategy: str = 'densest_packing',
        with_batch_norm: bool = True,
        with_relu: bool = True,
        with_residual: bool = False,
        enable_bias: bool = False,
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize the grid_interdependence_layer.

        This layer models interdependence across a grid using customizable patch-based structures and packing strategies.

        Parameters
        ----------
        h : int
            The height of the input grid.
        w : int
            The width of the input grid.
        in_channel : int
            The number of input channels.
        out_channel : int
            The number of output channels.
        d : int, optional
            The depth of the input grid. Default is 1 (for 2D grids).
        width : int, optional
            The number of grid interdependence heads. Default is 1.
        name : str, optional
            The name of the layer. Default is 'grid_interdependence_layer'.
        patch_shape : str, optional
            The shape of the patch ('cuboid', 'cylinder', or 'sphere'). Default is 'cuboid'.
        p_h : int, optional
            The height of the patch.
        p_h_prime : int, optional
            The height of the inner patch (used for hierarchical packing).
        p_w : int, optional
            The width of the patch.
        p_w_prime : int, optional
            The width of the inner patch (used for hierarchical packing).
        p_d : int, optional
            The depth of the patch.
        p_d_prime : int, optional
            The depth of the inner patch (used for hierarchical packing).
        p_r : int, optional
            The radius of the patch (used for spherical or cylindrical patches).
        cd_h : int, optional
            The compression degree along the height dimension.
        cd_w : int, optional
            The compression degree along the width dimension.
        cd_d : int, optional
            The compression degree along the depth dimension. Default is 1.
        packing_strategy : str, optional
            The packing strategy used ('densest_packing' or other supported strategies). Default is 'densest_packing'.
        with_batch_norm : bool, optional
            If True, applies batch normalization after the output. Default is True.
        with_relu : bool, optional
            If True, applies ReLU activation after the output. Default is True.
        with_residual : bool, optional
            If True, includes residual connections. Default is False.
        enable_bias : bool, optional
            If True, adds bias to the layer's weights. Default is False.
        with_dual_lphm : bool, optional
            If True, uses dual LPHM reconciliation for parameter fabrication. Default is False.
        with_lorr : bool, optional
            If True, uses LORR reconciliation for parameter fabrication. Default is False.
        r : int, optional
            The rank for parameter reconciliation methods. Default is 3.
        parameters_init_method : str, optional
            The initialization method for the parameters. Default is 'xavier_normal'.
        device : str, optional
            The device to use ('cpu' or 'cuda'). Default is 'cpu'.
        *args, **kwargs
            Additional arguments passed to the parent layer class.

        Raises
        ------
        ValueError
            If invalid patch shape or grid dimensions are provided.
        """
        print('* grid_interdependence_layer, width:', width)
        heads = [
            grid_interdependence_head(
                h=h, w=w, d=d,
                in_channel=in_channel, out_channel=out_channel,
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
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        assert len(heads) >= 1
        m, n = heads[0].get_m(), heads[0].get_n()
        if len(heads) > 1:
            head_fusion = mean_fusion(dims=[head.get_n() for head in heads])
        else:
            head_fusion = None
        print('--------------------------')
        super().__init__(name=name, m=m, n=n, heads=heads, head_fusion=head_fusion, device=device, *args, **kwargs)

    def get_output_grid_shape(self):
        """
        Get the shape of the output grid after packing.

        Returns
        -------
        tuple
            A tuple representing the dimensions of the output grid (height, width, depth).
        """
        assert len(self.heads) >= 1
        return self.heads[0].get_output_grid_shape()


class grid_compression_layer(layer):
    """
    A layer for compressing grid-based inputs using pooling metrics and patch-based aggregation.

    This layer supports operations to reduce the dimensionality of grid-based inputs by applying pooling metrics
    and packing strategies.

    Parameters
    ----------
    h : int
        The height of the input grid.
    w : int
        The width of the input grid.
    channel_num : int
        The number of input channels.
    d : int, optional
        The depth of the input grid. Default is 1 (2D grid).
    name : str, optional
        The name of the layer. Default is 'grid_compression_layer'.
    pooling_metric : str, optional
        The metric used for pooling ('batch_max', 'batch_min', etc.). Default is 'batch_max'.
    patch_shape : str, optional
        The shape of the patch ('cuboid', 'cylinder', or 'sphere'). Default is 'cuboid'.
    p_h : int, optional
        The height of the patch.
    p_h_prime : int, optional
        The height of the inner patch (for hierarchical packing).
    p_w : int, optional
        The width of the patch.
    p_w_prime : int, optional
        The width of the inner patch (for hierarchical packing).
    p_d : int, optional
        The depth of the patch (only for 3D grids).
    p_d_prime : int, optional
        The depth of the inner patch (for hierarchical packing).
    p_r : int, optional
        The radius of the patch (for spherical or cylindrical patches).
    cd_h : int, optional
        The compression degree along the height dimension.
    cd_w : int, optional
        The compression degree along the width dimension.
    cd_d : int, optional
        The compression degree along the depth dimension. Default is 1.
    with_dropout : bool, optional
        If True, applies dropout. Default is False.
    p : float, optional
        The dropout probability. Default is 0.5.
    packing_strategy : str, optional
        The packing strategy ('densest_packing' or other supported modes). Default is 'densest_packing'.
    parameters_init_method : str, optional
        The initialization method for parameters. Default is 'xavier_normal'.
    device : str, optional
        The device for computations ('cpu' or 'cuda'). Default is 'cpu'.
    *args, **kwargs
        Additional arguments passed to the parent class.

    Methods
    -------
    get_output_grid_shape():
        Returns the shape of the output grid after packing.

    Raises
    ------
    ValueError
        If invalid patch shape, dimensions, or configuration is provided.
    """
    def __init__(
        self,
        h: int, w: int, channel_num: int,
        d: int = 1,
        name: str = 'grid_compression_layer',
        pooling_metric: str = 'batch_max',
        patch_shape: str = 'cuboid',
        p_h: int = None, p_h_prime: int = None,
        p_w: int = None, p_w_prime: int = None,
        p_d: int = 0, p_d_prime: int = None,
        p_r: int = None,
        cd_h: int = None, cd_w: int = None, cd_d: int = 1,
        with_dropout: bool = False, p: float = 0.5,
        packing_strategy: str = 'densest_packing',
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize the grid_compression_layer.

        This layer compresses grid inputs by applying pooling metrics and packing strategies for dimensionality reduction.

        Parameters
        ----------
        h : int
            The height of the input grid.
        w : int
            The width of the input grid.
        channel_num : int
            The number of input channels.
        d : int, optional
            The depth of the input grid. Default is 1 (for 2D grids).
        name : str, optional
            The name of the layer. Default is 'grid_compression_layer'.
        pooling_metric : str, optional
            The metric used for pooling (e.g., 'batch_max', 'batch_min', 'batch_avg'). Default is 'batch_max'.
        patch_shape : str, optional
            The shape of the patch ('cuboid', 'cylinder', or 'sphere'). Default is 'cuboid'.
        p_h : int, optional
            The height of the patch.
        p_h_prime : int, optional
            The height of the inner patch (used for hierarchical packing).
        p_w : int, optional
            The width of the patch.
        p_w_prime : int, optional
            The width of the inner patch (used for hierarchical packing).
        p_d : int, optional
            The depth of the patch.
        p_d_prime : int, optional
            The depth of the inner patch (used for hierarchical packing).
        p_r : int, optional
            The radius of the patch (used for spherical or cylindrical patches).
        cd_h : int, optional
            The compression degree along the height dimension.
        cd_w : int, optional
            The compression degree along the width dimension.
        cd_d : int, optional
            The compression degree along the depth dimension. Default is 1.
        with_dropout : bool, optional
            If True, applies dropout after the compression step. Default is False.
        p : float, optional
            The dropout probability. Default is 0.5.
        packing_strategy : str, optional
            The packing strategy used ('densest_packing' or other supported strategies). Default is 'densest_packing'.
        parameters_init_method : str, optional
            The initialization method for the parameters. Default is 'xavier_normal'.
        device : str, optional
            The device to use ('cpu' or 'cuda'). Default is 'cpu'.
        *args, **kwargs
            Additional arguments passed to the parent layer class.

        Raises
        ------
        ValueError
            If invalid patch shape or grid dimensions are provided.
        """
        print('* grid_compression_layer')
        heads = [
            grid_compression_head(
                h=h, w=w, d=d,
                channel_num=channel_num,
                pooling_metric=pooling_metric,
                patch_shape=patch_shape,
                p_h=p_h, p_h_prime=p_h_prime,
                p_w=p_w, p_w_prime=p_w_prime,
                p_d=p_d, p_d_prime=p_d_prime,
                p_r=p_r,
                cd_h=cd_h, cd_w=cd_w, cd_d=cd_d,
                packing_strategy=packing_strategy,
                with_dropout=with_dropout, p=p,
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ]
        assert len(heads) >= 1
        m, n = heads[0].get_m(), heads[0].get_n()
        print('--------------------------')
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)

    def get_output_grid_shape(self):
        """
        Get the shape of the output grid after packing.

        Returns
        -------
        tuple
            A tuple representing the dimensions of the output grid (height, width, depth).
        """
        assert len(self.heads) >= 1
        return self.heads[0].get_output_grid_shape()

