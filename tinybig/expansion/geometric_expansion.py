# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################
# Geometric Expansion Function #
################################

from typing import Union
import torch

from tinybig.expansion import transformation
from tinybig.koala.geometry import grid as grid_structure
from tinybig.koala.geometry import cuboid, sphere, cylinder
from tinybig.interdependence.geometric_interdependence import geometric_interdependence


class geometric_expansion(transformation):
    r"""
        The geometric patch based expansion function.

        It performs the data expansion based on geometric patch shapes, where each attribute (or the attributes selected ones by the packing strategy) is expanded into a patch.
        This class inherits from the base data transformation class, and also implements the abstract methods in that class.

        ...

        Notes
        ----------
        Formally, given a data instance $\mathbf{x}$ and its underlying grid structure, we can extract a set of patches denoted as $\mathcal{P} = \{{p}_1, {p}_2, \cdots {p}_{|\mathcal{P}|}\}$.
        For simplicity, we use the notation $\mathbf{p}_i = \mathbf{x}(p_i) \in {R}^p$ to represent the attribute elements covered by patch ${p}_i \in \mathcal{P}$ from the input data instance vector $\mathbf{x}$.

        The geometric expansion function proposes to expand the input data instance with these patches as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \left[ \mathbf{p}_1, \mathbf{p}_2, \cdots, \mathbf{p}_{|\mathcal{P}|} \right] \in {R}^{d},
            \end{equation}
        $$

        where the expansion output vector dimension is $d = |\mathcal{P}| \times p$.

        The patches in set $\mathcal{P}$ can have different shapes, such as cuboid, cylinder and sphere.
        The size of the patch set $\mathcal{P}$ is determined by both the input grid shape, the patch shape and the packing strategies.

        Attributes
        ----------
        name: str, default = 'geometric_expansion'
            Name of the expansion function.
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
            It performs the initialization of the expansion function.

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
            It calculates the expansion space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method declared in the base transformation class.

    """

    def __init__(
        self,
        name: str = 'geometric_expansion',
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
            The initialization method of the geometric expansion function.

            It initializes the geometric expansion function based on
            the provided geometric grid (or its configuration or its size)
            and the patch shape (or the patch configuration) of the data segments to be compressed,
            and the packing strategy (or the patch center distances).

            Parameters
            ----------
            name: str, default = 'geometric_expansion'
                Name of the expansion function.
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
                The geometric expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

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
            The expansion dimension calculation method.

            It calculates the intermediate expansion space dimension based on the input dimension parameter m.
            For the geometric expansion function, the expansion space dimension is determined by
            the grid shape, patch shape and packing strategy.

            The expansion output vector dimension is $d = |\mathcal{P}| \times p$,
            where $\mathcal{P}$ is the patch set in the input and $p$ is the patch size.

            Parameters
            ----------
            m: int
                The dimension of the input space.

            Returns
            -------
            int
                The dimension of the expansion space.
        """
        assert m == self.get_grid_size(across_universe=True)
        return self.interdependence.get_patch_num(across_universe=True) * self.get_patch_size()

    def forward(self, x: torch.Tensor, device: str = 'cpu', *args, **kwargs) -> torch.Tensor:
        r"""
            The forward method of the geometric expansion function.

            It performs the geometric data expansion of the input data and returns the expansion result.

            Formally, given a data instance $\mathbf{x}$ and its underlying grid structure, we can extract a set of patches denoted as $\mathcal{P} = \{{p}_1, {p}_2, \cdots {p}_{|\mathcal{P}|}\}$.
            For simplicity, we use the notation $\mathbf{p}_i = \mathbf{x}(p_i) \in {R}^p$ to represent the attribute elements covered by patch ${p}_i \in \mathcal{P}$ from the input data instance vector $\mathbf{x}$.

            The geometric expansion function proposes to expand the input data instance with these patches as follows:

            $$
                \begin{equation}
                \kappa(\mathbf{x}) = \left[ \mathbf{p}_1, \mathbf{p}_2, \cdots, \mathbf{p}_{|\mathcal{P}|} \right] \in {R}^{d},
                \end{equation}
            $$

            where the expansion output vector dimension is $d = |\mathcal{P}| \times p$.

            Parameters
            ----------
            x: torch.Tensor
                The input data vector.
            device: str, default = 'cpu'
                The device of the input data vector.

            Returns
            -------
            torch.Tensor
                The expansion result.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        expansion = self.interdependence(x, device=device)
        expansion = expansion.view(b, self.get_patch_num(), -1)
        expansion = expansion.permute(0, 2, 1).reshape(b, -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class cuboid_patch_based_geometric_expansion(geometric_expansion):
    """
        The cuboid patch based geometric expansion function.

        It performs the data expansion based on cuboid geometric patch shapes and provided expansion metric.
        This class inherits from the geometric_expansion class, and only redefines the initialization method
        to declare the patch shape and size.

        ...

        Attributes
        ----------
        name: str, default = 'cuboid_patch_based_geometric_expansion'
            Name of the expansion function.
        p_h: int
            Height of the cuboid patch along the negative direction.
        p_h_prime: int, default = None
            Height of the cuboid patch along the positive direction.
        p_w: int
            Width of the cuboid patch along the negative direction.
        p_w_prime: int, default = None
            Width of the cuboid patch along the positive direction.
        p_d: int
            Depth of the cuboid patch along the negative direction.
        p_d_prime: int, default = None
            Depth of the cuboid patch along the positive direction.

        Methods
        ----------
        __init__
            It performs the initialization of the expansion function based on the provided metric and cuboid patch shapes.
    """
    def __init__(
        self,
        p_h: int, p_w: int, p_d: int,
        p_h_prime: int = None, p_w_prime: int = None, p_d_prime: int = None,
        name: str = 'cuboid_patch_based_geometric_expansion',
        *args, **kwargs
    ):
        """
            The initialization method of the cuboid patch based geometric expansion function.

            It initializes the cuboid expansion function based on the provided cuboid patch shape
            of the data segments to be compressed in the provided grid structure.

            Parameters
            ----------
            name: str, default = 'cuboid_patch_based_geometric_expansion'
                Name of the expansion function.
            p_h: int
                Height of the cuboid patch along the negative direction.
            p_h_prime: int, default = None
                Height of the cuboid patch along the positive direction.
            p_w: int
                Width of the cuboid patch along the negative direction.
            p_w_prime: int, default = None
                Width of the cuboid patch along the positive direction.
            p_d: int
                Depth of the cuboid patch along the negative direction.
            p_d_prime: int, default = None
                Depth of the cuboid patch along the positive direction.

            Returns
            ----------
            transformation
                The cuboid geometric expansion function.
        """
        patch = cuboid(p_h=p_h, p_w=p_w, p_d=p_d, p_h_prime=p_h_prime, p_w_prime=p_w_prime, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class cylinder_patch_based_geometric_expansion(geometric_expansion):
    """
        The cylinder patch based geometric expansion function.

        It performs the data expansion based on cylinder geometric patch shapes and provided expansion metric.
        This class inherits from the geometric_expansion class, and only redefines the initialization method
        to declare the patch shape and size.

        ...

        Attributes
        ----------
        name: str, default = 'cylinder_patch_based_geometric_expansion'
            Name of the expansion function.
        p_r: int
            Radius of the circular surface of the cylinder shape.
        p_d: int
            Depth of the cylinder patch along the negative direction.
        p_d_prime: int, default = None
            Depth of the cylinder patch along the positive direction.

        Methods
        ----------
        __init__
            It performs the initialization of the expansion function based on the provided metric and cylinder patch shapes.
    """
    def __init__(
        self,
        p_r: int, p_d: int, p_d_prime: int = None,
        name: str = 'cylinder_patch_based_geometric_expansion',
        *args, **kwargs
    ):
        """
            The initialization method of the cylinder patch based geometric expansion function.

            It initializes the cylinder expansion function based on the provided cylinder patch shape
            of the data segments to be compressed in the provided grid structure.

            Parameters
            ----------
            name: str, default = 'cylinder_patch_based_geometric_expansion'
                Name of the expansion function.
            p_r: int
                Radius of the circular surface of the cylinder shape.
            p_d: int
                Depth of the cylinder patch along the negative direction.
            p_d_prime: int, default = None
                Depth of the cylinder patch along the positive direction.

            Returns
            ----------
            transformation
                The cylinder geometric expansion function.
        """
        patch = cylinder(p_r=p_r, p_d=p_d, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class sphere_patch_based_geometric_expansion(geometric_expansion):
    """
        The sphere patch based geometric expansion function.

        It performs the data expansion based on sphere geometric patch shapes and provided expansion metric.
        This class inherits from the geometric_expansion class, and only redefines the initialization method
        to declare the patch shape and size.

        ...

        Attributes
        ----------
        name: str, default = 'sphere_patch_based_geometric_expansion'
            Name of the expansion function.
        p_r: int
            Radius of the sphere patch shape.

        Methods
        ----------
        __init__
            It performs the initialization of the expansion function based on the provided sphere patch shapes.
    """
    def __init__(
        self,
        p_r: int,
        name: str = 'sphere_patch_based_geometric_expansion',
        *args, **kwargs
    ):
        """
            The initialization method of the sphere patch based geometric expansion function.

            It initializes the sphere expansion function based on the provided sphere patch shape
            of the data segments to be compressed in the provided grid structure.

            Parameters
            ----------
            name: str, default = 'sphere_patch_based_geometric_expansion'
                Name of the expansion function.
            p_r: int
                Radius of the sphere shape.

            Returns
            ----------
            transformation
                The sphere geometric expansion function.
        """
        patch = sphere(p_r=p_r)
        super().__init__(name=name, patch=patch, *args, **kwargs)

