# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################
# Sphere Geometric Structure #
##############################

import math
import warnings

from tinybig.koala.geometry import geometric_space, coordinate_3d


class sphere(geometric_space):
    """
        Represents a spherical geometry in a 3D space.

        This class defines a sphere centered at a specific coordinate in a 3D space
        with configurable radius and packing strategies.

        Attributes
        ----------
        p_r : int
            Radius of the sphere.
        center : coordinate_3d
            The center coordinate of the sphere.
        name : str
            The name of the spherical geometry.
    """

    def __init__(
        self,
        p_r: int,
        center: coordinate_3d = coordinate_3d(0, 0, 0),
        name: str = 'sphere_geometry',
        *args, **kwargs
    ):
        """
            Initializes the spherical geometry.

            Parameters
            ----------
            p_r : int
                Radius of the sphere.
            center : coordinate_3d, optional
                The center coordinate of the sphere (default is coordinate_3d(0, 0, 0)).
            name : str, optional
                The name of the spherical geometry (default is 'sphere_geometry').
            *args, **kwargs
                Additional arguments for customization.
        """
        self.p_r = p_r
        super().__init__(center=center, name=name, *args, **kwargs)

    def radius(self):
        """
            Returns the radius of the sphere.

            Returns
            -------
            int
                The radius of the sphere.
        """
        return self.p_r

    @staticmethod
    def get_packing_strategies():
        """
            Returns the available packing strategies for the sphere.

            Returns
            -------
            list
                A list of packing strategies:
                ['sparse_simple_cubic', 'complete_simple_cubic', 'dense_simple_cubic',
                 'sparse_face_centered_cubic', 'complete_face_centered_cubic', 'dense_face_centered_cubic',
                 'sparse_hexagonal', 'complete_hexagonal', 'dense_hexagonal', 'densest_packing'].
        """
        return [
            'sparse_simple_cubic', 'complete_simple_cubic', 'dense_simple_cubic',
            'sparse_face_centered_cubic', 'complete_face_centered_cubic', 'dense_face_centered_cubic',
            'sparse_hexagonal', 'complete_hexagonal', 'dense_hexagonal',
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
                - 'sparse_simple_cubic': Centers are 2 times the radius apart.
                - 'complete_simple_cubic': Centers are 0.6667 * sqrt(3) times the radius apart.
                - 'dense_simple_cubic': Centers are equal to the radius apart.
                - 'sparse_face_centered_cubic': Centers are 2, sqrt(3), and 0.6667 * sqrt(6) times the radius apart.
                - 'complete_face_centered_cubic': Centers are sqrt(2), 0.6667 * sqrt(6), and 1.3333 times the radius apart.
                - 'dense_face_centered_cubic': Centers are equal to the radius apart.
                - 'sparse_hexagonal': Centers are 2 times the radius apart.
                - 'complete_hexagonal': Centers are 0.6667 * sqrt(3) times the radius apart.
                - 'dense_hexagonal': Centers are equal to the radius apart.
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
        if packing_strategy == 'sparse_simple_cubic':
            cd_h, cd_w, cd_d = 2 * self.p_r, 2 * self.p_r, 2 * self.p_r
        elif packing_strategy == 'complete_simple_cubic':
            cd_h, cd_w, cd_d = 0.6667 * math.sqrt(3) * self.p_r, 0.6667 * math.sqrt(3) * self.p_r, 0.6667 * math.sqrt(3) * self.p_r
        elif packing_strategy == 'dense_simple_cubic':
            cd_h, cd_w, cd_d = self.p_r, self.p_r, self.p_r

        elif packing_strategy == 'sparse_face_centered_cubic':
            cd_h, cd_w, cd_d = 2 * self.p_r, math.sqrt(3) * self.p_r, 0.6667 * math.sqrt(6) * self.p_r
        elif packing_strategy == 'complete_face_centered_cubic':
            cd_h, cd_w, cd_d = math.sqrt(2) * self.p_r, 0.6667 * math.sqrt(6) * self.p_r, 1.3333 * self.p_r
        elif packing_strategy == 'dense_face_centered_cubic':
            cd_h, cd_w, cd_d = self.p_r, self.p_r, self.p_r

        elif packing_strategy == 'sparse_hexagonal':
            cd_h, cd_w, cd_d = 2 * self.p_r, 2 * self.p_r, math.sqrt(6) * self.p_r
        elif packing_strategy == 'complete_hexagonal':
            cd_h, cd_w, cd_d = 0.6667 * math.sqrt(3)  * self.p_r, 0.6667 * math.sqrt(3) * self.p_r, self.p_r
        elif packing_strategy == 'dense_hexagonal':
            cd_h, cd_w, cd_d = self.p_r, self.p_r, self.p_r

        elif packing_strategy == 'densest_packing':
            cd_h, cd_w, cd_d = 1, 1, 1
        else:
            warnings.warn(f'Unknown strategy {packing_strategy}, will use the default densest_packing strategy...')
            cd_h, cd_w, cd_d = 1, 1, 1

        cd_h, cd_w, cd_d = max(int(cd_h), 1), max(int(cd_w), 1), max(int(cd_d), 1)
        return cd_h, cd_w, cd_d

    def generate_coordinates(self, *args, **kwargs):
        """
            Generates the coordinates of all points within the sphere.

            Parameters
            ----------
            *args, **kwargs
                Additional arguments for customization.

            Returns
            -------
            dict
                A dictionary where keys are coordinates within the sphere
                and values are indicators (default is 1).

            Notes
            -----
            This method uses the equation of a sphere:
            x^2 + y^2 + z^2 <= r^2
            to determine valid coordinates within the sphere.
        """
        coordinates = {}
        for i in range(-self.p_r, self.p_r+1):
            j_lim = int(math.sqrt(self.p_r ** 2 - i**2))
            for j in range(-j_lim, j_lim+1):
                # pre filtering
                if i ** 2 + j ** 2 <= self.p_r ** 2:
                    k_lim = int(math.sqrt(self.p_r ** 2 - i**2 - j**2))
                    for k in range(-k_lim, k_lim+1):
                        if i ** 2 + j ** 2 + k**2 <= self.p_r ** 2:
                            coord = coordinate_3d(i, j, k)
                            coordinates[coord + self.center] = 1
        return coordinates
