# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################
# Cylinder Geometric Structure #
################################

import math
import warnings

from tinybig.koala.geometry import geometric_space, coordinate_3d


class cylinder(geometric_space):
    """
        Represents a cylindrical geometry in a 3D space.

        This class defines a cylinder centered at a specific coordinate in a 3D space
        with configurable radius, depth, and packing strategies.

        Attributes
        ----------
        p_r : int
            Radius of the cylinder's base.
        p_d : int
            Depth extending backward from the center along the cylinder's axis.
        p_d_prime : int
            Depth extending forward from the center along the cylinder's axis.
        center : coordinate_3d
            The center coordinate of the cylinder.
        name : str
            The name of the cylindrical geometry.
    """
    def __init__(
        self,
        p_r: int, p_d: int = 0, p_d_prime: int = None,
        center: coordinate_3d = coordinate_3d(0, 0, 0),
        name: str = 'cylinder_geometry',
        *args, **kwargs
    ):
        """
            Initializes the cylindrical geometry.

            Parameters
            ----------
            p_r : int
                Radius of the cylinder's base.
            p_d : int, optional
                Depth extending backward from the center along the cylinder's axis (default is 0).
            p_d_prime : int, optional
                Depth extending forward from the center along the cylinder's axis (default is equal to `p_d`).
            center : coordinate_3d, optional
                The center coordinate of the cylinder (default is coordinate_3d(0, 0, 0)).
            name : str, optional
                The name of the cylindrical geometry (default is 'cylinder_geometry').
            *args, **kwargs
                Additional arguments for customization.
        """
        self.p_r = p_r
        self.p_d = p_d
        self.p_d_prime = p_d_prime if p_d_prime is not None else p_d
        super().__init__(center=center, name=name, *args, **kwargs)

    def radius(self):
        """
            Returns the radius of the cylinder's base.

            Returns
            -------
            int
                The radius of the cylinder's base.
        """
        return self.p_r

    def depth(self):
        """
            Calculates the total depth of the cylinder.

            Returns
            -------
            int
                The total depth of the cylinder, including both forward and backward extensions.
        """
        return self.p_d + self.p_d_prime + 1

    @staticmethod
    def get_packing_strategies():
        """
            Returns the available packing strategies for the cylinder.

            Returns
            -------
            list
                A list of packing strategies:
                ['sparse_square', 'complete_square', 'sparse_hexagonal',
                 'complete_hexagonal', 'dense_hexagonal', 'denser_hexagonal', 'densest_packing'].
        """
        return [
            'sparse_square', 'complete_square',
            'sparse_hexagonal', 'complete_hexagonal', 'dense_hexagonal', 'denser_hexagonal',
            'densest_packing',
        ]

    def packing_strategy_parameters(self, packing_strategy: str = 'complete_square', *args, **kwargs):
        """
            Determines the center distances for packing based on the selected strategy.

            Parameters
            ----------
            packing_strategy : str, optional
                The packing strategy to use (default is 'complete_square').
                Options include:
                - 'sparse_square': Centers are 2 times the radius apart.
                - 'complete_square': Centers are sqrt(2) times the radius apart.
                - 'sparse_hexagonal': Centers are sqrt(3) times the radius apart.
                - 'complete_hexagonal': Centers are 1.5 times and sqrt(3) times the radius apart.
                - 'dense_hexagonal': Centers are sqrt(6)/2 and sqrt(2) times the radius apart.
                - 'denser_hexagonal': Centers are sqrt(3)/2 and the radius apart.
                - 'densest_packing': Centers are 1 unit apart.

            *args, **kwargs
                Additional arguments for customization.

            Returns
            -------
            tuple
                A tuple (cd_h, cd_w, cd_d) representing the center distances for height, width, and depth.

            Raises
            ------
            ValueError
                If an unknown packing strategy is provided.
        """
        if packing_strategy == 'sparse_square':
            cd_h, cd_w, cd_d = 2 * self.p_r, 2 * self.p_r, 2 * self.p_d
        elif packing_strategy == 'complete_square':
            cd_h, cd_w, cd_d = math.sqrt(2) * self.p_r, math.sqrt(2) * self.p_r, 2 * self.p_d

        elif packing_strategy == 'sparse_hexagonal':
            cd_h, cd_w, cd_d = math.sqrt(3) * self.p_r, 2 * self.p_r, 2 * self.p_d
        elif packing_strategy == 'complete_hexagonal':
            cd_h, cd_w, cd_d = 1.5 * self.p_r, math.sqrt(3) * self.p_r, 2 * self.p_d
        elif packing_strategy == 'dense_hexagonal':
            cd_h, cd_w, cd_d = 0.5 * math.sqrt(6) * self.p_r, math.sqrt(2) * self.p_r, self.p_d
        elif packing_strategy == 'denser_hexagonal':
            cd_h, cd_w, cd_d = 0.5 * math.sqrt(3) * self.p_r, self.p_r, self.p_d

        elif packing_strategy == 'densest_packing':
            cd_h, cd_w, cd_d = 1, 1, 1
        else:
            warnings.warn(f'Unknown strategy {packing_strategy}, will use the default densest_packing strategy...')
            cd_h, cd_w, cd_d = 1, 1, 1

        cd_h, cd_w, cd_d = max(int(cd_h), 1), max(int(cd_w), 1), max(int(cd_d), 1)
        return cd_h, cd_w, cd_d

    def generate_coordinates(self, *args, **kwargs):
        """
            Generates the coordinates of all points within the cylinder.

            Parameters
            ----------
            *args, **kwargs
                Additional arguments for customization.

            Returns
            -------
            dict
                A dictionary where keys are coordinates within the cylinder
                and values are indicators (default is 1).
        """
        coordinates = {}
        for i in range(-self.p_r, self.p_r+1):
            j_lim = int(math.sqrt(self.p_r ** 2 - i ** 2))
            for j in range(-j_lim, j_lim + 1):
                if i**2 + j**2 <= self.p_r**2:
                    for k in range(-self.p_d_prime, self.p_d_prime+1):
                        coord = coordinate_3d(i, j, k)
                        coordinates[coord + self.center] = 1
        return coordinates
