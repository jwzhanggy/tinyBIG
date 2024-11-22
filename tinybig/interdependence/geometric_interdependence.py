# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############################
# Geometric Interdependence #
#############################
"""
The geometric interdependence functions

This module contains the geometric interdependence functions, including
    geometric_interdependence,
    cuboid_patch_based_geometric_interdependence,
    cylinder_patch_based_geometric_interdependence,
    sphere_patch_based_geometric_interdependence,
    cuboid_patch_padding_based_geometric_interdependence,
    cuboid_patch_aggregation_based_geometric_interdependence,
    cylinder_patch_padding_based_geometric_interdependence,
    cylinder_patch_aggregation_based_geometric_interdependence,
    sphere_patch_padding_based_geometric_interdependence,
    sphere_patch_aggregation_based_geometric_interdependence,
"""

from typing import Union
import torch

from tinybig.interdependence import interdependence
from tinybig.koala.geometry import grid as grid_structure
from tinybig.koala.geometry import cuboid, sphere, cylinder
from tinybig.config.base_config import config


class geometric_interdependence(interdependence):
    r"""
        A class for geometric interdependence function.

        This class represents interdependence relationships based on a grid and patch structure.
        It supports various packing strategies and interdependence matrix processing options.

        Notes
        ----------

        __Grid Structure__

        Formally, given a data instance vector $\mathbf{x} \in R^{m}$, we can represent its underlying 3D grid structure
        as an ordered list of coordinates denoting the attributes' locations in the grid:
        $$
            \begin{equation}
            grid(\mathbf{x} | h, w, d) = \left[ \left(i, j, k\right) \right]_{i \in \{0, 1, \cdots, h-1\}, j \in \{0, 1, \cdots, w-1\}, k \in \{0, 1, \cdots, d-1\}}.
            \end{equation}
        $$

        __Patch Shapes__

        Given a coordinate tuple $(i, j, k)$ in the grid, we can represent the patch, e.g., a cuboid with shape $(p_h, p_h'; p_w, p_w'; p_d, p_d')$, centered at $(i, j, k)$ as an ordered list of coordinate tuples:
        $$
            \begin{equation}
            patch(i, j, k) = \left[ \left( i+ \nabla i, j + \nabla j, k + \nabla k \right) \right]_{\nabla i \in [-p_h, p_h'], \nabla j \in [- p_w, p_w'], \nabla k \in [- p_d, p_d']}.
            \end{equation}
        $$

        Similarly, we can represent cylinder patches of shape $(r; p_d, p_d')$ and sphere patches of shape $(r)$ centered at coordinates $(i, j, k)$ as follows:
        $$
            \begin{equation}
            patch(i, j, k) = \left[ \left( i+ \nabla i, j + \nabla j, k + \nabla k \right) \right]_{\nabla i, \nabla j \in [-r, r] \land \nabla i^2 + \nabla j^2 \le r^2, \nabla k \in [- p_d, p_d']},
            \end{equation}
        $$

        and

        $$
            \begin{equation}
            patch(i, j, k) = \left[ \left( i+ \nabla i, j + \nabla j, k + \nabla k \right) \right]_{\nabla i, \nabla j, \nabla k \in [-r, r] \land \nabla i^2 + \nabla j^2 + \nabla k^2 \le r^2}.
            \end{equation}
        $$
        whose size is also represented by the term $p = \left| patch(i, j, k) \right|$ by default.

        __Operation Modes__

        The geometric interdependence function can operate in both the padding and aggregation modes.

        In the interdependence padding mode, the function composes matrix $\mathbf{A}$ as the concatenation of a sequence of block matrices:

        $$
            \begin{equation}
            \xi(\mathbf{x}) = \mathbf{A} = \left[ \mathbf{A}_{(i, j, k)} \right]_{(i,j,k) \in grid(\mathbf{x} | h, w, d)} \in R^{m \times m'}.
            \end{equation}
        $$

        For each coordinate tuple $(i, j, k) \in grid(\mathbf{x} | h, w, d)$ in the underlying grid structure of instance vector $\mathbf{x}$,
        a block sub-matrix $\mathbf{A}_{(i,j,k)} \in R^{m \times p}$ is defined.

        In contrast, the interdependence matrix defined in the aggregation mode is considerably denser:
        $$
            \begin{equation}
            \xi(\mathbf{x}) = \mathbf{A} \in R^{m \times m'}.
            \end{equation}
        $$

        In the underlying grid structure of instance vector $\mathbf{x}$, each coordinate tuple $(i, j, k) \in grid(\mathbf{x} | h, w, d)$
        corresponds to a specific column in matrix $\mathbf{A}$. This column is uniquely identified by the index $idx(i,j,k)$.

        Attributes
        ----------
        grid : grid_structure
            The grid structure used for interdependence calculation.
        patch : Union[cuboid, cylinder, sphere]
            The patch structure used for interdependence.
        packing_strategy : str
            Strategy for packing the patches ('densest_packing', etc.).
        interdependence_matrix_mode : str
            Mode for processing the interdependence matrix ('padding' or 'aggregation').
        normalization : bool
            If True, normalizes the interdependence matrix.
        normalization_mode : str
            Mode of normalization ('row_column', etc.).
        cd_h, cd_w, cd_d : int
            Packing parameters for height, width, and depth of the patches.

        Methods
        -------
        __init__(...)
            Initializes the geometric interdependence function.
        update_grid(new_grid)
            Updates the grid structure.
        update_packing_strategy(new_packing_strategy)
            Updates the packing strategy and adjusts packing parameters.
        update_patch(new_patch)
            Updates the patch structure and adjusts packing parameters.
        update_packing_parameters(new_cd_h, new_cd_w, new_cd_d)
            Updates the packing parameters.
        get_channel_num()
            Returns the number of channels in the grid.
        get_patch_size()
            Returns the volume of the patch.
        get_patch_num(across_universe=False)
            Returns the number of patches in the grid.
        get_grid_size(across_universe=False)
            Returns the volume of the grid.
        get_grid_shape()
            Returns the shape of the grid.
        get_grid_shape_after_packing()
            Returns the shape of the grid after packing.
        calculate_b_prime(b=None, across_universe=True)
            Computes the number of rows in the output tensor after interdependence.
        calculate_m_prime(m=None, across_universe=True)
            Computes the number of columns in the output tensor after interdependence.
        calculate_A(...)
            Computes the interdependence matrix.
        forward(...)
            Applies interdependence transformation to the input tensor.
    """
    def __init__(
        self,
        b: int = 0, m: int = 0,
        interdependence_type: str = 'attribute',
        name: str = 'geometric_interdependence',
        # grid structure initialization options
        grid: grid_structure = None,
        grid_configs: dict = None,
        h: int = None, w: int = None, d: int = 1, channel_num: int = 1,
        # patch structure initialization options
        patch: Union[cuboid, cylinder, sphere] = None,
        patch_configs: dict = None,
        # packing options
        packing_strategy: str = 'densest_packing',
        cd_h: int = None, cd_w: int = None, cd_d: int = None,
        interdependence_matrix_mode: str = 'padding',
        # interdependence matrix processing options
        normalization: bool = False,
        normalization_mode: str = 'row_column',
        # by default,
        require_data: bool = False, require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        """
            Initializes the geometric interdependence function.

            Parameters
            ----------
            b : int, optional
                Number of rows in the input tensor. Defaults to 0.
            m : int, optional
                Number of columns in the input tensor. Defaults to 0.
            interdependence_type : str, optional
                Type of interdependence ('attribute', 'instance', etc.). Defaults to 'attribute'.
            name : str, optional
                Name of the interdependence function. Defaults to 'geometric_interdependence'.
            grid : grid_structure, optional
                Grid structure for interdependence. Defaults to None.
            grid_configs : dict, optional
                Configuration dictionary for grid initialization. Defaults to None.
            h, w, d : int, optional
                Dimensions of the grid (height, width, depth). Defaults to None.
            channel_num : int, optional
                Number of channels in the grid. Defaults to 1.
            patch : Union[cuboid, cylinder, sphere], optional
                Patch structure for interdependence. Defaults to None.
            patch_configs : dict, optional
                Configuration dictionary for patch initialization. Defaults to None.
            packing_strategy : str, optional
                Packing strategy for patches. Defaults to 'densest_packing'.
            cd_h, cd_w, cd_d : int, optional
                Packing parameters (height, width, depth). Defaults to None.
            interdependence_matrix_mode : str, optional
                Mode for interdependence matrix processing ('padding' or 'aggregation'). Defaults to 'padding'.
            normalization : bool, optional
                If True, normalizes the interdependence matrix. Defaults to False.
            normalization_mode : str, optional
                Mode of normalization ('row_column', etc.). Defaults to 'row_column'.
            require_data : bool, optional
                If True, requires input data for matrix calculation. Defaults to False.
            require_parameters : bool, optional
                If True, requires parameters for matrix calculation. Defaults to False.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

            Raises
            ------
            ValueError
                If the grid or patch structure is not specified.
                If the interdependence type is not supported.
        """
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data, require_parameters=require_parameters, device=device, *args, **kwargs)

        if grid is not None:
            self.grid = grid
        elif grid_configs is not None:
            self.grid = config.instantiation_from_configs(configs=grid_configs, class_name='grid_class', parameter_name='grid_parameters')
        elif h is not None and w is not None and d is not None:
            grid_parameters = {'h': h, 'w': w, 'd': d, 'universe_num': channel_num}
            self.grid = grid_structure(**grid_parameters)
        else:
            raise ValueError('the grid structure is not specified yet...')

        if interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            if self.m is None or self.m <= 0:
                self.m = self.grid.get_volume(across_universe=True)
            assert self.grid.get_volume(across_universe=True) == self.m
        elif interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            if self.b is None or self.b <= 0:
                self.b = self.grid.get_volume(across_universe=True)
            assert self.grid.get_volume(across_universe=True) == self.b
        else:
            raise ValueError('the interdependence_type is not supported yet...')

        if patch is not None:
            self.patch = patch
        elif patch_configs is not None:
            self.patch = config.instantiation_from_configs(configs=patch_configs, class_name='patch_class', parameter_name='patch_parameters')
        else:
            raise ValueError('the patch structure is not specified yet...')

        self.packing_strategy = packing_strategy

        self.cd_h, self.cd_w, self.cd_d = (cd_h, cd_w, cd_d) if (cd_h is not None and cd_w is not None and cd_d is not None) else self.patch.packing_strategy_parameters(packing_strategy=self.packing_strategy)

        self.interdependence_matrix_mode = interdependence_matrix_mode
        self.normalization = normalization
        self.normalization_mode = normalization_mode

    def update_grid(self, new_grid: grid_structure):
        """
            Updates the grid structure.

            Parameters
            ----------
            new_grid : grid_structure
                The new grid structure to replace the current one.
        """
        self.grid = new_grid

    def update_packing_strategy(self, new_packing_strategy: str):
        """
            Updates the packing strategy and adjusts packing parameters.

            Parameters
            ----------
            new_packing_strategy : str
                The new packing strategy to use.
        """
        self.packing_strategy = new_packing_strategy
        self.cd_h, self.cd_w, self.cd_d = self.patch.packing_strategy_parameters(packing_strategy=self.packing_strategy)

    def update_patch(self, new_patch: Union[cuboid, cylinder, sphere]):
        """
            Updates the patch structure and adjusts packing parameters.

            Parameters
            ----------
            new_patch : Union[cuboid, cylinder, sphere]
                The new patch structure to replace the current one.
        """
        self.patch = new_patch
        self.cd_h, self.cd_w, self.cd_d = self.patch.packing_strategy_parameters(packing_strategy=self.packing_strategy)

    def update_packing_parameters(self, new_cd_h: int, new_cd_w: int, new_cd_d: int):
        """
            Updates the packing parameters.

            Parameters
            ----------
            new_cd_h : int
                New packing parameter for height.
            new_cd_w : int
                New packing parameter for width.
            new_cd_d : int
                New packing parameter for depth.
        """
        self.cd_h = new_cd_h
        self.cd_w = new_cd_w
        self.cd_d = new_cd_d

    def get_channel_num(self):
        """
            Returns the number of channels in the grid structure.

            This corresponds to the number of universes in the grid.

            Returns
            -------
            int
                The number of channels in the grid.
        """
        return self.grid.get_universe_num()

    def get_patch_size(self):
        """
            Returns the size (volume) of the patch.

            The size is calculated as the number of elements in the patch.

            Returns
            -------
            int
                The volume of the patch.
        """
        return self.patch.get_volume()

    def get_patch_num(self, across_universe: bool = False):
        """
            Returns the number of patches in the grid.

            Parameters
            ----------
            across_universe : bool, optional
                If True, computes the number of patches across all universes. Defaults to False.

            Returns
            -------
            int
                The total number of patches in the grid.
        """
        return self.grid.get_patch_num(cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d, across_universe=across_universe)

    def get_grid_size(self, across_universe: bool = False):
        """
            Returns the size (volume) of the grid.

            Parameters
            ----------
            across_universe : bool, optional
                If True, computes the size across all universes. Defaults to False.

            Returns
            -------
            int
                The total volume of the grid.
        """
        return self.grid.get_volume(across_universe=across_universe)

    def get_grid_shape(self):
        """
            Returns the shape of the grid.

            Returns
            -------
            tuple of int
                A tuple representing the dimensions of the grid (height, width, depth).
        """
        return self.grid.get_grid_shape()

    def get_grid_shape_after_packing(self):
        """
            Returns the shape of the grid after applying the packing strategy.

            The shape is adjusted based on the current packing parameters.

            Returns
            -------
            tuple of int
                A tuple representing the packed dimensions of the grid.
        """
        return self.grid.get_grid_shape_after_packing(cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d)

    def calculate_b_prime(self, b: int = None, across_universe: bool = True):
        """
            Computes the number of rows in the output tensor after applying interdependence function.

            Parameters
            ----------
            b : int, optional
                Number of rows in the input tensor. If None, defaults to `self.b`.
            across_universe : bool, optional
                If True, computes the result across all universes. Defaults to True.

            Returns
            -------
            int
                The number of rows in the output tensor.

            Raises
            ------
            AssertionError
                If the input dimensions do not match the grid structure.
        """
        b = b if b is not None else self.b
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert self.grid.get_volume(across_universe=across_universe) == b
            if self.interdependence_matrix_mode == 'padding':
                return self.grid.get_patch_num(cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d, across_universe=across_universe) * self.patch.get_volume()
            elif self.interdependence_matrix_mode == 'aggregation':
                return b
        else:
            return b

    def calculate_m_prime(self, m: int = None, across_universe: bool = True):
        """
            Computes the number of columns in the output tensor after applying interdependence.

            Parameters
            ----------
            m : int, optional
                Number of columns in the input tensor. If None, defaults to `self.m`.
            across_universe : bool, optional
                If True, computes the result across all universes. Defaults to True.

            Returns
            -------
            int
                The number of columns in the output tensor.

            Raises
            ------
            AssertionError
                If the input dimensions do not match the grid structure.
        """
        m = m if m is not None else self.m
        if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert self.grid.get_volume(across_universe=across_universe) == m
            if self.interdependence_matrix_mode == 'padding':
                return self.grid.get_patch_num(cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d, across_universe=across_universe) * self.patch.get_volume()
            elif self.interdependence_matrix_mode == 'aggregation':
                return m
        else:
            return m

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, across_universe: bool = False, device: str = 'cpu', *args, **kwargs):
        """
            Computes the interdependence matrix.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter matrix for computation. Defaults to None.
            across_universe : bool, optional
                If True, computes the interdependence across all universes. Defaults to False.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The computed interdependence matrix.

            Raises
            ------
            ValueError
                If required data or parameters are not provided.
        """

        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            A = self.grid.to_matrix(
                patch=self.patch, packing_strategy=self.packing_strategy,
                cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d,
                interdependence_matrix_mode=self.interdependence_matrix_mode,
                normalization=self.normalization, normalization_mode=self.normalization_mode,
                across_universe=across_universe, device=device, *args, **kwargs
            )
            A = self.post_process(x=A, device=device)
            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A

    def forward(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, kappa_x: torch.Tensor = None, device: str = 'cpu', *args, **kwargs):
        """
            Applies interdependence transformation to the input tensor.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Required if `self.require_data` is True. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter matrix for computation. Required if `self.require_parameters` is True. Defaults to None.
            kappa_x : torch.Tensor, optional
                A secondary input tensor. Defaults to None.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Transformed tensor after applying the interdependence function.

            Raises
            ------
            AssertionError
                If required inputs are not provided.
            ValueError
                If the interdependence type is invalid.
        """

        if self.require_data:
            assert x is not None and x.ndim == 2
        if self.require_parameters:
            assert w is not None and w.ndim == 2

        data_x = kappa_x if kappa_x is not None else x
        b, m = data_x.shape
        data_x = data_x.view(b*self.grid.get_universe_num(), -1)

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            # A shape: b * b'
            A = self.calculate_A(x.transpose(0, 1), w, device=device)

            assert A is not None and A.size(0) == data_x.size(0)
            if data_x.is_sparse or A.is_sparse:
                xi_x = torch.sparse.mm(A.t(), data_x)
            else:
                xi_x = torch.matmul(A.t(), data_x)

            return xi_x
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            # A shape: m * m'
            A = self.calculate_A(x, w, device=device)

            assert A is not None and A.size(0) == data_x.size(-1)
            if data_x.is_sparse or A.is_sparse:
                xi_x = torch.sparse.mm(data_x, A)
            else:
                xi_x = torch.matmul(data_x, A)

            if self.interdependence_matrix_mode == 'padding':
                # shape [b, c, g, p] -> shape [b, g, c, p]
                xi_x = xi_x.view(b, self.grid.get_universe_num(), self.get_patch_num(), self.get_patch_size())
                xi_x = xi_x.permute(0, 2, 1, 3)

            elif self.interdependence_matrix_mode == 'aggregation':
                # shape [b, c, g] -> shape [b, g, c]
                xi_x = xi_x.view(b, self.grid.get_universe_num(), self.get_patch_num())
                xi_x = xi_x.permute(0, 2, 1)

            return xi_x.reshape(b, -1)
        else:
            raise ValueError(f"Invalid interdependence type: {self.interdependence_type}")


class cuboid_patch_based_geometric_interdependence(geometric_interdependence):
    """
        A geometric interdependence class with cuboid patches.

        This class uses cuboid-shaped patches for interdependence calculations.

        Methods
        -------
        __init__(...)
            Initializes the cuboid patch-based geometric interdependence function.
    """
    def __init__(
        self,
        p_h: int, p_w: int, p_d: int,
        p_h_prime: int = None, p_w_prime: int = None, p_d_prime: int = None,
        name: str = 'cuboid_patch_based_geometric_interdependence',
        *args, **kwargs
    ):
        """
            Initializes the cuboid patch-based geometric interdependence function.

            Parameters
            ----------
            p_h, p_w, p_d : int
                Dimensions of the cuboid patch (height, width, depth).
            p_h_prime, p_w_prime, p_d_prime : int, optional
                Output dimensions of the cuboid patch. Defaults to None.
            name : str, optional
                Name of the interdependence function. Defaults to 'cuboid_patch_based_geometric_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        patch = cuboid(p_h=p_h, p_w=p_w, p_d=p_d, p_h_prime=p_h_prime, p_w_prime=p_w_prime, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class cuboid_patch_padding_based_geometric_interdependence(cuboid_patch_based_geometric_interdependence):
    """
        A geometric interdependence class using cuboid patches with padding mode.

        This class applies geometric interdependence in padding mode, ensuring patches are padded
        into the grid structure.

        Attributes
        ----------
        interdependence_matrix_mode : str
            The interdependence matrix mode set to 'padding'.

        Methods
        -------
        __init__(interdependence_matrix_mode='padding', ...)
            Initializes the cuboid patch-based geometric interdependence function in padding mode.
    """
    def __init__(self, interdependence_matrix_mode: str = 'padding', *args, **kwargs):
        """
            Initializes the cuboid patch-based geometric interdependence function in padding mode.

            Parameters
            ----------
            interdependence_matrix_mode : str, optional
                Mode for the interdependence matrix. Defaults to 'padding'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(interdependence_matrix_mode='padding', *args, **kwargs)


class cuboid_patch_aggregation_based_geometric_interdependence(cuboid_patch_based_geometric_interdependence):
    """
        A geometric interdependence class using cuboid patches with aggregation mode.

        This class applies geometric interdependence in aggregation mode, ensuring patches are
        aggregated into the grid structure.

        Attributes
        ----------
        interdependence_matrix_mode : str
            The interdependence matrix mode set to 'aggregation'.

        Methods
        -------
        __init__(interdependence_matrix_mode='aggregation', ...)
            Initializes the cuboid patch-based geometric interdependence function in aggregation mode.
    """
    def __init__(self, interdependence_matrix_mode: str = 'aggregation', *args, **kwargs):
        """
            Initializes the cuboid patch-based geometric interdependence function in aggregation mode.

            Parameters
            ----------
            interdependence_matrix_mode : str, optional
                Mode for the interdependence matrix. Defaults to 'aggregation'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(interdependence_matrix_mode='aggregation', *args, **kwargs)


class cylinder_patch_based_geometric_interdependence(geometric_interdependence):
    """
        A geometric interdependence class using cylindrical patches.

        This class applies geometric interdependence based on cylinder-shaped patches within a grid structure.

        Attributes
        ----------
        patch : cylinder
            The cylinder patch structure used for interdependence.

        Methods
        -------
        __init__(r, p_d, p_d_prime=None, ...)
            Initializes the cylindrical patch-based geometric interdependence function.
    """

    def __init__(
        self,
        r: int, p_d: int, p_d_prime: int = None,
        name: str = 'cylinder_patch_based_geometric_interdependence',
        *args, **kwargs
    ):
        """
            Initializes the cylindrical patch-based geometric interdependence function.

            Parameters
            ----------
            r : int
                Radius of the cylinder patch.
            p_d : int
                Depth of the cylinder patch.
            p_d_prime : int, optional
                Output depth of the cylinder patch. Defaults to None.
            name : str, optional
                Name of the interdependence function. Defaults to 'cylinder_patch_based_geometric_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        patch = cylinder(r=r, p_d=p_d, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class cylinder_patch_padding_based_geometric_interdependence(cylinder_patch_based_geometric_interdependence):
    """
        A geometric interdependence class using cylindrical patches with padding mode.

        Attributes
        ----------
        interdependence_matrix_mode : str
            The interdependence matrix mode set to 'padding'.

        Methods
        -------
        __init__(interdependence_matrix_mode='padding', ...)
            Initializes the cylindrical patch-based geometric interdependence function in padding mode.
    """
    def __init__(self, interdependence_matrix_mode: str = 'padding', *args, **kwargs):
        """
            Initializes the cylindrical patch-based geometric interdependence function in padding mode.

            Parameters
            ----------
            interdependence_matrix_mode : str, optional
                Mode for the interdependence matrix. Defaults to 'padding'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(interdependence_matrix_mode='padding', *args, **kwargs)


class cylinder_patch_aggregation_based_geometric_interdependence(cylinder_patch_based_geometric_interdependence):
    """
        A geometric interdependence class using cylindrical patches with aggregation mode.

        This class applies geometric interdependence in aggregation mode, ensuring patches are aggregated
        into the grid structure.

        Attributes
        ----------
        interdependence_matrix_mode : str
            The interdependence matrix mode set to 'aggregation'.

        Methods
        -------
        __init__(interdependence_matrix_mode='aggregation', ...)
            Initializes the cylindrical patch-based geometric interdependence function in aggregation mode.
    """
    def __init__(self, interdependence_matrix_mode: str = 'aggregation', *args, **kwargs):
        """
            Initializes the cylindrical patch-based geometric interdependence function in aggregation mode.

            Parameters
            ----------
            interdependence_matrix_mode : str, optional
                Mode for the interdependence matrix. Defaults to 'aggregation'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(interdependence_matrix_mode='aggregation', *args, **kwargs)


class sphere_patch_based_geometric_interdependence(geometric_interdependence):
    """
        A geometric interdependence class using spherical patches.

        This class applies geometric interdependence based on sphere-shaped patches within a grid structure.

        Attributes
        ----------
        patch : sphere
            The sphere patch structure used for interdependence.

        Methods
        -------
        __init__(r, ...)
            Initializes the spherical patch-based geometric interdependence function.
    """
    def __init__(
        self,
        r: int,
        name: str = 'sphere_patch_based_geometric_interdependence',
        *args, **kwargs
    ):
        """
            Initializes the spherical patch-based geometric interdependence function.

            Parameters
            ----------
            r : int
                Radius of the sphere patch.
            name : str, optional
                Name of the interdependence function. Defaults to 'sphere_patch_based_geometric_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        patch = sphere(r=r)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class sphere_patch_padding_based_geometric_interdependence(sphere_patch_based_geometric_interdependence):
    """
        A geometric interdependence class using spherical patches with padding mode.

        This class applies geometric interdependence in padding mode, ensuring patches are padded into
        the grid structure.

        Attributes
        ----------
        interdependence_matrix_mode : str
            The interdependence matrix mode set to 'padding'.

        Methods
        -------
        __init__(interdependence_matrix_mode='padding', ...)
            Initializes the spherical patch-based geometric interdependence function in padding mode.
    """
    def __init__(self, interdependence_matrix_mode: str = 'padding', *args, **kwargs):
        """
            Initializes the spherical patch-based geometric interdependence function in padding mode.

            Parameters
            ----------
            interdependence_matrix_mode : str, optional
                Mode for the interdependence matrix. Defaults to 'padding'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(interdependence_matrix_mode='padding', *args, **kwargs)


class sphere_patch_aggregation_based_geometric_interdependence(sphere_patch_based_geometric_interdependence):
    """
        A geometric interdependence class using spherical patches with aggregation mode.

        This class applies geometric interdependence in aggregation mode, ensuring patches are aggregated
        into the grid structure.

        Attributes
        ----------
        interdependence_matrix_mode : str
            The interdependence matrix mode set to 'aggregation'.

        Methods
        -------
        __init__(interdependence_matrix_mode='aggregation', ...)
            Initializes the spherical patch-based geometric interdependence function in aggregation mode.
    """
    def __init__(self, interdependence_matrix_mode: str = 'aggregation', *args, **kwargs):
        """
            Initializes the spherical patch-based geometric interdependence function in aggregation mode.

            Parameters
            ----------
            interdependence_matrix_mode : str, optional
                Mode for the interdependence matrix. Defaults to 'aggregation'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(interdependence_matrix_mode='aggregation', *args, **kwargs)

