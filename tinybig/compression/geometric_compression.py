# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Geometric Compression Function #
##################################

"""
Geometric data compression functions.

This module contains the geometric data compression functions,
including geometric_compression, and its variants with different geometric patch shapes and compression metrics.
"""

from typing import Union, Callable
import torch

from tinybig.compression import transformation
from tinybig.koala.linear_algebra import batch_max, batch_min
from tinybig.koala.statistics import batch_mean
from tinybig.koala.geometry import grid as grid_structure
from tinybig.koala.geometry import cuboid, sphere, cylinder
from tinybig.interdependence.geometric_interdependence import geometric_interdependence


class geometric_compression(transformation):
    r"""
        The geometric patch based compression function.

        It performs the data compression based on geometric patch shapes and provided compression metric.
        This class inherits from the base data transformation class, and also implements the abstract methods in that class.

        ...

        Notes
        ----------
        Formally, given a data instance $\mathbf{x}$ and its underlying grid structure, we can extract a set of patches denoted as $\mathcal{P} = \{{p}_1, {p}_2, \cdots {p}_{|\mathcal{P}|}\}$.
        For simplicity, we use the notation $\mathbf{p}_i = \mathbf{x}(p_i) \in {R}^p$ to represent the attribute elements covered by patch ${p}_i \in \mathcal{P}$ from the input data instance vector $\mathbf{x}$.

        The geometric compression function proposes to compress the patch vector $\mathbf{p}_i$ using a mapping $\phi: {R}^p \to {R}^{d_{\phi}}$, which transforms it into a dense representation of length $d_{\phi}$ as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \left[ \phi(\mathbf{p}_1), \phi(\mathbf{p}_2), \cdots, \phi(\mathbf{p}_{|\mathcal{P}|}) \right] \in {R}^{d},
            \end{equation}
        $$

        where the compression output vector dimension is $d = |\mathcal{P}| \times d_{\phi}$.
        The dimension parameter $d_{\phi}$ must be manually specified when defining the patch vector compression mapping $\phi$.
        For the majority of mappings $\phi$ studied in this project, the output is typically a scalar, i.e., the dimension $d_{\phi} = 1$.

        The patches in set $\mathcal{P}$ can have different shapes, such as cuboid, cylinder and sphere.
        The compression metric $\phi$ can also have different definitions, such as max, min and mean.

        Attributes
        ----------
        metric: Callable[[torch.Tensor], torch.Tensor]
            The geometric compression metric.
        name: str, default = 'geometric_compression'
            Name of the compression function.
        grid: grid, default = None
            The input data instance underlying grid structure.
        grid_configs: dict, default = None
            The grid detailed configurations.
        h: int, default = None
            The height of the grid structure.
        w: int, default = None
            The width of the grid structure.
        d: int, default = None
            The depth of the grid structure.
        patch: Union[cuboid, cylinder, sphere], default = None
            The patch shape of the data segments to be compressed.
        patch_configs: dict, default = None
            The patch detailed configurations.
        packing_strategy: str, default = None
            The packing strategy to be used.
        cd_h: int, default = None
            The patch center distance along the height dimension.
        cd_w: int, default = None
            The patch center distance along the width dimension.
        cd_d: int, default = None
            The patch center distance along the depth dimension.

        Methods
        ----------
        __init__
            It performs the initialization of the compression function.

        get_grid_shape
            It returns the shape of the grid structure.

        get_grid_size
            It returns the size of the grid structure.

        get_patch_size
            It returns the size of the patch structure.

        get_patch_num
            It returns the number of patches in the input data vector.

        get_grid_shape_after_packing
            It returns the shape of the grid structure after packing, i.e., the patch center coordinates.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method declared in the base transformation class.

    """

    def __init__(
        self,
        metric: Callable[[torch.Tensor], torch.Tensor],
        name: str = 'geometric_compression',
        grid: grid_structure = None,
        grid_configs: dict = None,
        h: int = None, w: int = None, d: int = None,
        patch: Union[cuboid, cylinder, sphere] = None,
        patch_configs: dict = None,
        packing_strategy: str = None,
        cd_h: int = None, cd_w: int = None, cd_d: int = None,
        *args, **kwargs
    ):
        """
            The initialization method of the geometric compression function.

            It initializes the geometric compression function based on
            the provided geometric grid (or its configuration or its size)
            and the patch shape (or the patch configuration) of the data segments to be compressed,
            and the packing strategy (or the patch center distances).

            Parameters
            ----------
            metric: Callable[[torch.Tensor], torch.Tensor]
                The geometric compression metric.
            name: str, default = 'geometric_compression'
                Name of the compression function.
            grid: grid, default = None
                The input data instance underlying grid structure.
            grid_configs: dict, default = None
                The grid detailed configurations.
            h: int, default = None
                The height of the grid structure.
            w: int, default = None
                The width of the grid structure.
            d: int, default = None
                The depth of the grid structure.
            patch: Union[cuboid, cylinder, sphere], default = None
                The patch shape of the data segments to be compressed.
            patch_configs: dict, default = None
                The patch detailed configurations.
            packing_strategy: str, default = None
                The packing strategy to be used.
            cd_h: int, default = None
                The patch center distance along the height dimension.
            cd_w: int, default = None
                The patch center distance along the width dimension.
            cd_d: int, default = None
                The patch center distance along the depth dimension.

            Returns
            ----------
            transformation
                The geometric compression function.
        """
        super().__init__(name=name, *args, **kwargs)

        self.metric = metric

        self.interdependence = geometric_interdependence(
            b=0, m=0, interdependence_type='attribute', interdependence_matrix_mode='padding',
            grid=grid, grid_configs=grid_configs, h=h, w=w, d=d,
            patch=patch, patch_configs=patch_configs,
            packing_strategy=packing_strategy, cd_h=cd_h, cd_w=cd_w, cd_d=cd_d,
            normalization=False, require_data=False, require_parameters=False,
        )

    def get_grid_shape(self):
        """
            The grid shape retrieval function.

            It returns the shape of the grid structure.

            Returns
            -------
            tuple | list
                The shape of the grid structure.
        """
        return self.interdependence.get_grid_shape()

    def get_grid_size(self, across_universe: bool = False):
        """
            The grid shape retrieval function.

            It returns the size of the grid structure.

            Parameters
            ----------
            across_universe: bool, default = False
                The boolean tag indicating the grid size across universe or not.

            Returns
            -------
            int
                The size of the grid structure.
        """
        return self.interdependence.get_grid_size(across_universe=across_universe)

    def get_patch_size(self):
        """
            The patch shape retrieval function.

            It returns the size of the patch structure.

            Returns
            -------
            int
                The size of the patch structure.
        """
        return self.interdependence.get_patch_size()

    def get_patch_num(self, across_universe: bool = False):
        """
            The patch shape number function.

            It returns the number of patches existing in the input data vector.

            Parameters
            ----------
            across_universe: bool, default = False
                The boolean tag indicating the patch size across universe or not.

            Returns
            -------
            int
                The number of patches existing in the input data vector.
        """
        return self.interdependence.get_patch_num(across_universe=across_universe)

    def get_grid_shape_after_packing(self):
        """
            The shape of the grid structure after packing.

            It returns the shape of the grid structure after packing.

            Returns
            -------
            tuple | list
                The shape of the grid structure after packing.
        """
        return self.interdependence.get_grid_shape_after_packing()

    def calculate_D(self, m: int):
        r"""
            The compression dimension calculation method.

            It calculates the intermediate compression space dimension based on the input dimension parameter m.
            For the geometric compression function, the compression space dimension is determined by
            the grid shape, patch shape and packing strategy.

            The compression output vector dimension is $d = |\mathcal{P}| \times d_{\phi}$,
            where $\mathcal{P}$ is the patch set in the input and
            $d_{\phi}$ denotes the compression output dimension of the metric $\phi$.

            Parameters
            ----------
            m: int
                The dimension of the input space.

            Returns
            -------
            int
                The dimension of the compression space.
        """
        assert m == self.get_grid_size(across_universe=True)
        return self.interdependence.get_patch_num(across_universe=True)

    def forward(self, x: torch.Tensor, device: str = 'cpu', *args, **kwargs) -> torch.Tensor:
        r"""
            The forward method of the geometric compression function.

            It performs the geometric data compression of the input data and returns the compression result.

            Formally, given a data instance $\mathbf{x}$ and its underlying grid structure, we can extract a set of patches denoted as $\mathcal{P} = \{{p}_1, {p}_2, \cdots {p}_{|\mathcal{P}|}\}$.
            For simplicity, we use the notation $\mathbf{p}_i = \mathbf{x}(p_i) \in {R}^p$ to represent the attribute elements covered by patch ${p}_i \in \mathcal{P}$ from the input data instance vector $\mathbf{x}$.

            The geometric compression function proposes to compress the patch vector $\mathbf{p}_i$ using a mapping $\phi: {R}^p \to {R}^{d_{\phi}}$, which transforms it into a dense representation of length $d_{\phi}$ as follows:

            $$
                \begin{equation}
                \kappa(\mathbf{x}) = \left[ \phi(\mathbf{p}_1), \phi(\mathbf{p}_2), \cdots, \phi(\mathbf{p}_{|\mathcal{P}|}) \right] \in {R}^{d},
                \end{equation}
            $$

            where the compression output vector dimension is $d = |\mathcal{P}| \times d_{\phi}$.

            Parameters
            ----------
            x: torch.Tensor
                The input data vector.
            device: str, default = 'cpu'
                The device of the input data vector.

            Returns
            -------
            torch.Tensor
                The compression result.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        x = self.interdependence(x, device=device)

        p = self.get_patch_size()
        compression = self.metric(x.view(-1, p), dim=1).view(b, self.get_patch_num(), -1)
        compression = compression.permute(0, 2, 1).reshape(b, -1)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class cuboid_patch_based_geometric_compression(geometric_compression):
    """
        The geometric patch based compression function.

        ...

        Notes
        ----------

        Attributes
        ----------

        Methods
        ----------

    """
    def __init__(
        self,
        p_h: int, p_w: int, p_d: int,
        p_h_prime: int = None, p_w_prime: int = None, p_d_prime: int = None,
        name: str = 'cuboid_patch_based_geometric_compression',
        *args, **kwargs
    ):
        patch = cuboid(p_h=p_h, p_w=p_w, p_d=p_d, p_h_prime=p_h_prime, p_w_prime=p_w_prime, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class cylinder_patch_based_geometric_compression(geometric_compression):

    def __init__(
        self,
        p_r: int, p_d: int, p_d_prime: int = None,
        name: str = 'cylinder_patch_based_geometric_compression',
        *args, **kwargs
    ):
        patch = cylinder(p_r=p_r, p_d=p_d, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class sphere_patch_based_geometric_compression(geometric_compression):

    def __init__(
        self,
        p_r: int,
        name: str = 'sphere_patch_based_geometric_compression',
        *args, **kwargs
    ):

        patch = sphere(p_r=p_r)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class cuboid_max_based_geometric_compression(cuboid_patch_based_geometric_compression):
    def __init__(self, name: str = 'cuboid_max_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class cuboid_min_based_geometric_compression(cuboid_patch_based_geometric_compression):
    def __init__(self, name: str = 'cuboid_min_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class cuboid_mean_based_geometric_compression(cuboid_patch_based_geometric_compression):
    def __init__(self, name: str = 'cuboid_mean_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)


class cylinder_max_based_geometric_compression(cylinder_patch_based_geometric_compression):
    def __init__(self, name: str = 'cylinder_max_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class cylinder_min_based_geometric_compression(cylinder_patch_based_geometric_compression):
    def __init__(self, name: str = 'cylinder_min_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class cylinder_mean_based_geometric_compression(cylinder_patch_based_geometric_compression):
    def __init__(self, name: str = 'cylinder_mean_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)


class sphere_max_based_geometric_compression(sphere_patch_based_geometric_compression):
    def __init__(self, name: str = 'sphere_max_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class sphere_min_based_geometric_compression(sphere_patch_based_geometric_compression):
    def __init__(self, name: str = 'sphere_min_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class sphere_mean_based_geometric_compression(sphere_patch_based_geometric_compression):
    def __init__(self, name: str = 'sphere_mean_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)
