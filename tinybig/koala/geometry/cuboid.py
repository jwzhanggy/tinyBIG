# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################
# Cuboid Geometric Structure #
##############################

import warnings

from tinybig.koala.geometry import geometric_space, coordinate_3d


class cuboid(geometric_space):
    """
        Represents a cuboid geometry in a 3D space.

        This class defines a cuboid centered at a specific coordinate in a 3D space
        with configurable dimensions for height, width, and depth, along with packing strategies.

        Attributes
        ----------
        p_h : int
            Number of units extending upward from the center along the height.
        p_w : int
            Number of units extending sideways from the center along the width.
        p_d : int
            Number of units extending backward from the center along the depth.
        p_h_prime : int
            Number of units extending downward from the center along the height.
        p_w_prime : int
            Number of units extending the opposite direction along the width.
        p_d_prime : int
            Number of units extending forward from the center along the depth.
        center : coordinate_3d
            The center coordinate of the cuboid.
        name : str
            The name of the cuboid geometry.
    """
    def __init__(
        self,
        p_h: int, p_w: int, p_d: int = 0,
        p_h_prime: int = None, p_w_prime: int = None, p_d_prime: int = None,
        center: coordinate_3d = coordinate_3d(0, 0, 0),
        name: str = 'cuboid_geometry',
        *args, **kwargs
    ):
        """
            Initializes the cuboid geometry.

            Parameters
            ----------
            p_h : int
                Number of units extending upward from the center along the height.
            p_w : int
                Number of units extending sideways from the center along the width.
            p_d : int, optional
                Number of units extending backward from the center along the depth (default is 0).
            p_h_prime : int, optional
                Number of units extending downward from the center along the height (default is equal to `p_h`).
            p_w_prime : int, optional
                Number of units extending the opposite direction along the width (default is equal to `p_w`).
            p_d_prime : int, optional
                Number of units extending forward from the center along the depth (default is equal to `p_d`).
            center : coordinate_3d, optional
                The center coordinate of the cuboid (default is coordinate_3d(0, 0, 0)).
            name : str, optional
                The name of the cuboid geometry (default is 'cuboid_geometry').
            *args, **kwargs
                Additional arguments for customization.
        """
        self.p_h = p_h
        self.p_w = p_w
        self.p_d = p_d
        self.p_h_prime = p_h_prime if p_h_prime is not None else p_h
        self.p_w_prime = p_w_prime if p_w_prime is not None else p_w
        self.p_d_prime = p_d_prime if p_d_prime is not None else p_d
        super().__init__(center=center, name=name, *args, **kwargs)

    def shape(self):
        """
            Returns the dimensions of the cuboid as a tuple.

            Returns
            -------
            tuple
                A tuple containing the height, width, and depth of the cuboid.
        """
        return self.height(), self.width(), self.depth()

    def height(self):
        """
            Calculates the height of the cuboid.

            Returns
            -------
            int
                The height of the cuboid.
        """
        return self.p_h + self.p_h_prime + 1

    def width(self):
        """
            Calculates the width of the cuboid.

            Returns
            -------
            int
                The width of the cuboid.
        """
        return self.p_w + self.p_w_prime + 1

    def depth(self):
        """
            Calculates the depth of the cuboid.

            Returns
            -------
            int
                The depth of the cuboid.
        """
        return self.p_d + self.p_d_prime + 1

    @staticmethod
    def get_packing_strategies():
        """
            Returns the available packing strategies for the cuboid.

            Returns
            -------
            list
                A list of packing strategies: ['sparse_square', 'complete_square', 'dense_square', 'densest_packing'].
        """
        return ['sparse_square', 'complete_square', 'dense_square', 'densest_packing']

    def packing_strategy_parameters(self, packing_strategy: str = 'complete_square', *args, **kwargs):
        """
            Determines the center distances for packing based on the selected strategy.

            Parameters
            ----------
            packing_strategy : str, optional
                The packing strategy to use (default is 'complete_square').
                Options include:
                - 'sparse_square': Centers are 3 times the dimensions apart.
                - 'complete_square': Centers are 2 times the dimensions apart.
                - 'dense_square': Centers are 1 time the dimensions apart.
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
            cd_h, cd_w, cd_d = 3 * self.p_h, 3 * self.p_w, 3 * self.p_d
        elif packing_strategy == 'complete_square':
            cd_h, cd_w, cd_d = 2 * self.p_h, 2 * self.p_w, 2 * self.p_d
        elif packing_strategy == 'dense_square':
            cd_h, cd_w, cd_d = self.p_h, self.p_w, self.p_d

        elif packing_strategy == 'densest_packing':
            cd_h, cd_w, cd_d = 1, 1, 1
        else:
            warnings.warn(f'Unknown strategy {packing_strategy}, will use the default densest_packing strategy...')
            cd_h, cd_w, cd_d = 1, 1, 1

        cd_h, cd_w, cd_d = max(int(cd_h), 1), max(int(cd_w), 1), max(int(cd_d), 1)
        return cd_h, cd_w, cd_d

    def generate_coordinates(self, *args, **kwargs):
        """
            Generates the coordinates of all points within the cuboid.

            Parameters
            ----------
            *args, **kwargs
                Additional arguments for customization.

            Returns
            -------
            dict
                A dictionary where keys are coordinates within the cuboid
                and values are indicators (default is 1).
        """
        coordinates = {}
        for i in range(-self.p_h, self.p_h_prime+1):
            for j in range(-self.p_w, self.p_w_prime+1):
                for k in range(-self.p_d, self.p_d_prime+1):
                    coord = coordinate_3d(i, j, k)
                    coordinates[coord + self.center] = 1
        return coordinates
