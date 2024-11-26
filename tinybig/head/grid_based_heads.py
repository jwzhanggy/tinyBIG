# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################
# Grid Based Head Modules #
###########################

"""
Grid Structural RPN based heads.

This module contains the grid structural rpn based heads, including
    grid_interdependence_head

"""

from functools import partial

import torch

from tinybig.module.base_head import head
from tinybig.koala.linear_algebra.metric import metric
from tinybig.koala.algebra import find_close_factors
from tinybig.expansion.basic_expansion import identity_expansion
from tinybig.interdependence.geometric_interdependence import geometric_interdependence
from tinybig.compression.geometric_compression import geometric_compression
from tinybig.reconciliation.basic_reconciliation import constant_eye_reconciliation, identity_reconciliation
from tinybig.reconciliation.lowrank_reconciliation import lorr_reconciliation, dual_lphm_reconciliation, hm_reconciliation, lphm_reconciliation
from tinybig.remainder.basic_remainder import zero_remainder, linear_remainder
from tinybig.koala.geometry import grid, cuboid, cylinder, sphere


class grid_interdependence_head(head):
    """
    A head for handling grid-based interdependence mechanisms.

    This class supports different patch structures (cuboid, cylinder, sphere) and geometric interdependence.
    It includes parameter reconciliation, data transformation, and output processing.

    Attributes
    ----------
    h : int
        Height of the grid.
    w : int
        Width of the grid.
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    d : int, default=1
        Depth of the grid.
    name : str
        Name of the head.
    patch_shape : str, default='cuboid'
        Shape of the patch. Options: 'cuboid', 'cylinder', 'sphere'.
    cd_h, cd_w, cd_d : int, optional
        Parameters for packing and compression.
    packing_strategy : str, default='densest_packing'
        Strategy for packing grid patches.
    with_batch_norm : bool, default=True
        Whether to apply batch normalization.
    with_relu : bool, default=True
        Whether to apply ReLU activation.
    with_residual : bool, default=False
        Whether to include a residual connection.
    with_dual_lphm : bool, default=False
        Whether to use dual low-rank parameterized hyper-matrix (LPHM) for parameter reconciliation.
    with_lorr : bool, default=False
        Whether to use LORR (Low-rank Orthogonal Reconciliation).
    r : int, default=3
        Rank for parameter reconciliation.
    enable_bias : bool, default=False
        Whether to enable bias.
    parameters_init_method : str, default='xavier_normal'
        Initialization method for parameters.
    device : str, default='cpu'
        Device to run the computations ('cpu' or 'cuda').

    Methods
    -------
    get_patch_size()
        Returns the size of the patch.
    get_input_grid_shape()
        Returns the shape of the input grid.
    get_output_grid_shape()
        Returns the shape of the output grid after packing.
    calculate_phi_w(channel_index, device, *args, **kwargs)
        Computes the phi_w parameter using parameter fabrication.
    calculate_inner_product(kappa_xi_x, phi_w, device, *args, **kwargs)
        Calculates the inner product of the given tensors.
    """
    def __init__(
        self,
        h: int, w: int, in_channel: int, out_channel: int,
        d: int = 1, name: str = 'grid_interdependence_head',
        # patch structure parameters
        patch_shape: str = 'cuboid',
        p_h: int = None, p_h_prime: int = None,
        p_w: int = None, p_w_prime: int = None,
        p_d: int = 0, p_d_prime: int = None,
        p_r: int = None,
        # packing parameters
        cd_h: int = None, cd_w: int = None, cd_d: int = 1,
        packing_strategy: str = 'densest_packing',
        # output processing function parameters
        with_batch_norm: bool = True,
        with_relu: bool = True,
        with_residual: bool = False,
        # other parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = False,
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the `grid_interdependence_head` class.

        Parameters
        ----------
        h : int
            Height of the grid.
        w : int
            Width of the grid.
        in_channel : int
            Number of input channels.
        out_channel : int
            Number of output channels.
        d : int, default=1
            Depth of the grid.
        name : str, default='grid_interdependence_head'
            Name of the head.
        patch_shape : str, default='cuboid'
            Shape of the patch. Options: 'cuboid', 'cylinder', 'sphere'.
        p_h : int, optional
            Patch height.
        p_h_prime : int, optional
            Adjusted patch height for the cuboid.
        p_w : int, optional
            Patch width. Defaults to `p_h` if not provided.
        p_w_prime : int, optional
            Adjusted patch width for the cuboid.
        p_d : int, default=0
            Patch depth.
        p_d_prime : int, optional
            Adjusted patch depth for the cuboid.
        p_r : int, optional
            Patch radius (for spherical or cylindrical patches).
        cd_h : int, optional
            Compression depth in the height dimension.
        cd_w : int, optional
            Compression depth in the width dimension.
        cd_d : int, default=1
            Compression depth in the depth dimension.
        packing_strategy : str, default='densest_packing'
            Strategy for packing patches into the grid.
        with_batch_norm : bool, default=True
            Whether to apply batch normalization to the output.
        with_relu : bool, default=True
            Whether to apply ReLU activation to the output.
        with_residual : bool, default=False
            Whether to include a residual connection.
        with_dual_lphm : bool, default=False
            Whether to use dual low-rank parameterized hyper-matrix (LPHM) reconciliation.
        with_lorr : bool, default=False
            Whether to use low-rank orthogonal reconciliation (LORR).
        r : int, default=3
            Rank used for parameter reconciliation.
        enable_bias : bool, default=False
            Whether to enable bias in the linear transformations.
        parameters_init_method : str, default='xavier_normal'
            Initialization method for model parameters.
        device : str, default='cpu'
            Device to run the computations ('cpu' or 'cuda').
        """
        if in_channel is None or out_channel is None or in_channel <=0 or out_channel <=0:
            raise ValueError(f'positive in_channel={in_channel} and out_channel={out_channel} must be specified...')
        self.in_channel = in_channel
        self.out_channel = out_channel

        if h is None or w is None or d is None:
            raise ValueError(f'h={h} and w={w} and d={d} must be specified...')
        grid_structure = grid(
            h=h, w=w, d=d, universe_num=in_channel
        )

        if patch_shape == 'cuboid':
            assert p_h is not None
            p_w = p_w if p_w is not None else p_h
            patch_structure = cuboid(p_h=p_h, p_w=p_w, p_d=p_d, p_h_prime=p_h_prime, p_w_prime=p_w_prime, p_d_prime=p_d_prime)
        elif patch_shape == 'cylinder':
            assert p_r is not None
            patch_structure = cylinder(p_r=p_r, p_d=p_d, p_d_prime=p_d_prime)
        elif patch_shape == 'sphere':
            assert p_r is not None
            patch_structure = sphere(p_r=p_r)
        else:
            raise ValueError(f'patch_shape={patch_shape} must be either cuboid, cylinder or sphere...')

        attribute_interdependence = geometric_interdependence(
            interdependence_type='attribute',
            grid=grid_structure,
            patch=patch_structure,
            packing_strategy=packing_strategy,
            cd_h=cd_h, cd_w=cd_w, cd_d=cd_d,
            interdependence_matrix_mode='padding',
            normalization=False,
            require_data=False, require_parameters=False,
            device=device
        )

        data_transformation = identity_expansion(
            device=device
        )

        if with_dual_lphm:
            print('grid head', 'with_dual_lphm:', with_dual_lphm, 'r:', r)
            parameter_fabrication = dual_lphm_reconciliation(
                r=r,
                device=device,
                enable_bias=enable_bias,
            )
        elif with_lorr:
            print('grid head', 'with_lorr:', with_lorr, 'r:', r)
            parameter_fabrication = lorr_reconciliation(
                r=r,
                device=device,
                enable_bias=enable_bias,
            )
        else:
            parameter_fabrication = identity_reconciliation(
                device=device,
                enable_bias=enable_bias
            )
        # to save computational cost, the n we provide the parameter fabrication function is different from the n of the head,
        # we need to manually provide the l for the parameter fabrication functions...
        l = parameter_fabrication.calculate_l(
            n=self.out_channel, D=self.in_channel*attribute_interdependence.get_patch_size()
        )

        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )

        m = attribute_interdependence.get_grid_size(across_universe=True)
        n = attribute_interdependence.get_patch_num(across_universe=False) * out_channel

        output_process_functions = []
        if with_batch_norm:
            output_process_functions.append(torch.nn.BatchNorm1d(num_features=n, device=device))
        if with_relu:
            output_process_functions.append(torch.nn.ReLU())

        print('conv layer', output_process_functions)

        super().__init__(
            name=name,
            m=m, n=n, channel_num=1, l=l,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            attribute_interdependence=attribute_interdependence,
            output_process_functions=output_process_functions,
            parameters_init_method=parameters_init_method,
            device=device, *args, **kwargs
        )

    def get_patch_size(self):
        """
        Returns the size of the patch in the grid.

        Returns
        -------
        int
            Patch size.
        """
        return self.attribute_interdependence.get_patch_size()

    def get_input_grid_shape(self):
        """
        Returns the shape of the input grid.

        Returns
        -------
        tuple
            A tuple representing (height, width, depth) of the input grid.
        """
        return self.attribute_interdependence.get_grid_shape()

    def get_output_grid_shape(self):
        """
        Returns the shape of the output grid after packing.

        Returns
        -------
        tuple
            A tuple representing (height, width, depth) of the packed grid.
        """
        return self.attribute_interdependence.get_grid_shape_after_packing()

    def calculate_phi_w(self, channel_index: int = 0, device='cpu', *args, **kwargs):
        """
        Computes the phi_w parameter using parameter fabrication.

        Parameters
        ----------
        channel_index : int, default=0
            Index of the channel.
        device : str, default='cpu'
            Device to perform the computation.

        Returns
        -------
        torch.Tensor
            The phi_w tensor after parameter fabrication.
        """
        assert channel_index in range(self.channel_num)
        w_chunk = self.w[channel_index:channel_index + 1, :]
        n, D = self.out_channel, self.in_channel * self.attribute_interdependence.get_patch_size()
        assert w_chunk.size(1) == self.parameter_fabrication.calculate_l(n=n, D=D)
        phi_w = self.parameter_fabrication(w=w_chunk, n=n, D=D, device=device)
        return phi_w

    def calculate_inner_product(self, kappa_xi_x: torch.Tensor, phi_w: torch.Tensor, device='cpu', *args, **kwargs):
        """
        Calculates the inner product of the given tensors.

        Parameters
        ----------
        kappa_xi_x : torch.Tensor
            Input tensor for inner product calculation.
        phi_w : torch.Tensor
            Weight tensor for inner product calculation.
        device : str, default='cpu'
            Device to perform the computation.

        Returns
        -------
        torch.Tensor
            The computed inner product.
        """
        assert kappa_xi_x.ndim == 2 and phi_w.ndim == 2
        b = kappa_xi_x.size(0)
        inner_prod = torch.matmul(kappa_xi_x.view(b, -1, self.get_patch_size() * self.in_channel), phi_w.T)
        inner_prod = inner_prod.permute(0, 2, 1).reshape(b, -1)
        if self.b is not None:
            inner_prod += self.b
        return inner_prod


class grid_compression_head(head):
    """
    A head for compressing grid data using geometric compression.

    Supports different patch structures (cuboid, cylinder, sphere) and packing strategies for grid data.

    Attributes
    ----------
    h : int
        Height of the grid.
    w : int
        Width of the grid.
    channel_num : int
        Number of channels in the grid.
    d : int, default=1
        Depth of the grid.
    name : str
        Name of the head.
    pooling_metric : str, default='batch_max'
        Metric used for pooling operations.
    patch_shape : str, default='cuboid'
        Shape of the patch. Options: 'cuboid', 'cylinder', 'sphere'.
    cd_h, cd_w, cd_d : int, optional
        Parameters for packing and compression.
    packing_strategy : str, default='densest_packing'
        Strategy for packing grid patches.
    with_dropout : bool, default=True
        Whether to apply dropout during output processing.
    p : float, default=0.5
        Dropout probability.
    parameters_init_method : str, default='xavier_normal'
        Initialization method for parameters.
    device : str, default='cpu'
        Device to run the computations ('cpu' or 'cuda').

    Methods
    -------
    get_patch_size()
        Returns the size of the patch.
    get_input_grid_shape()
        Returns the shape of the input grid.
    get_output_grid_shape()
        Returns the shape of the output grid after packing.
    """
    def __init__(
        self,
        h: int, w: int, channel_num: int,
        d: int = 1, name: str = 'grid_compression_head',
        pooling_metric: str = 'batch_max',
        patch_shape: str = 'cuboid',
        p_h: int = None, p_h_prime: int = None,
        p_w: int = None, p_w_prime: int = None,
        p_d: int = 0, p_d_prime: int = None,
        p_r: int = None,
        cd_h: int = None, cd_w: int = None, cd_d: int = 1,
        packing_strategy: str = 'densest_packing',
        with_dropout: bool = True, p: float = 0.5,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the `grid_compression_head` class.

        Parameters
        ----------
        h : int
            Height of the grid.
        w : int
            Width of the grid.
        channel_num : int
            Number of channels in the grid.
        d : int, default=1
            Depth of the grid.
        name : str, default='grid_compression_head'
            Name of the head.
        pooling_metric : str, default='batch_max'
            Metric to use for pooling operations.
        patch_shape : str, default='cuboid'
            Shape of the patch. Options: 'cuboid', 'cylinder', 'sphere'.
        p_h : int, optional
            Patch height.
        p_h_prime : int, optional
            Adjusted patch height for the cuboid.
        p_w : int, optional
            Patch width. Defaults to `p_h` if not provided.
        p_w_prime : int, optional
            Adjusted patch width for the cuboid.
        p_d : int, default=0
            Patch depth.
        p_d_prime : int, optional
            Adjusted patch depth for the cuboid.
        p_r : int, optional
            Patch radius (for spherical or cylindrical patches).
        cd_h : int, optional
            Compression depth in the height dimension.
        cd_w : int, optional
            Compression depth in the width dimension.
        cd_d : int, default=1
            Compression depth in the depth dimension.
        packing_strategy : str, default='densest_packing'
            Strategy for packing patches into the grid.
        with_dropout : bool, default=True
            Whether to apply dropout during output processing.
        p : float, default=0.5
            Dropout probability.
        parameters_init_method : str, default='xavier_normal'
            Initialization method for model parameters.
        device : str, default='cpu'
            Device to run the computations ('cpu' or 'cuda').
        """
        if channel_num is None or channel_num <=0:
            raise ValueError(f'positive channel number={channel_num} must be specified...')
        self.channel_num = channel_num
        if h is None or w is None or d is None:
            raise ValueError(f'h={h} and w={w} and d={d} must be specified...')
        grid_structure = grid(
            h=h, w=w, d=d, universe_num=channel_num
        )

        if patch_shape == 'cuboid':
            assert p_h is not None
            p_w = p_w if p_w is not None else p_h
            patch_structure = cuboid(p_h=p_h, p_w=p_w, p_d=p_d, p_h_prime=p_h_prime, p_w_prime=p_w_prime, p_d_prime=p_d_prime)
        elif patch_shape == 'cylinder':
            assert p_r is not None
            patch_structure = cylinder(p_r=p_r, p_d=p_d, p_d_prime=p_d_prime)
        elif patch_shape == 'sphere':
            assert p_r is not None
            patch_structure = sphere(p_r=p_r)
        else:
            raise ValueError(f'patch_shape={patch_shape} must be either cuboid, cylinder or sphere...')

        data_transformation = geometric_compression(
            grid=grid_structure,
            patch=patch_structure,
            packing_strategy=packing_strategy,
            cd_h=cd_h, cd_w=cd_w, cd_d=cd_d,
            metric=partial(metric, metric_name=pooling_metric),
            device=device,
        )

        remainder = zero_remainder(
            device=device,
        )

        output_process_functions = []
        if with_dropout:
            output_process_functions.append(torch.nn.Dropout(p=p))
        print('pooling layer', output_process_functions)

        m = data_transformation.get_grid_size(across_universe=True)
        n = data_transformation.get_patch_num(across_universe=True)

        super().__init__(
            m=m, n=n,
            name=name,
            data_transformation=data_transformation,
            remainder=remainder,
            output_process_functions=output_process_functions,
            parameters_init_method=parameters_init_method,
            device=device, *args, **kwargs
        )

    def get_patch_size(self):
        """
        Returns the size of the patch in the grid.

        Returns
        -------
        int
            Patch size.
        """
        return self.data_transformation.get_patch_size()

    def get_input_grid_shape(self):
        """
        Returns the shape of the input grid.

        Returns
        -------
        tuple
            A tuple representing (height, width, depth) of the input grid.
        """
        return self.data_transformation.get_grid_shape()

    def get_output_grid_shape(self):
        """
        Returns the shape of the output grid after packing.

        Returns
        -------
        tuple
            A tuple representing (height, width, depth) of the packed grid.
        """
        output_h, output_w, output_d = self.data_transformation.get_grid_shape_after_packing()
        return output_h, output_w, output_d
