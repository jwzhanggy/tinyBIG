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

    def __init__(
        self,
        p_r: int, p_d: int = 0, p_d_prime: int = None,
        center: coordinate_3d = coordinate_3d(0, 0, 0),
        name: str = 'cylinder_geometry',
        *args, **kwargs
    ):
        self.p_r = p_r
        self.p_d = p_d
        self.p_d_prime = p_d_prime if p_d_prime is not None else p_d
        super().__init__(center=center, name=name, *args, **kwargs)

    def radius(self):
        return self.p_r

    def depth(self):
        return self.p_d + self.p_d_prime + 1

    @staticmethod
    def get_packing_strategies():
        return [
            'sparse_square', 'complete_square',
            'sparse_hexagonal', 'complete_hexagonal', 'dense_hexagonal', 'denser_hexagonal',
            'densest_packing',
        ]

    def packing_strategy_parameters(self, packing_strategy: str = 'complete_square', *args, **kwargs):

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
        coordinates = {}
        for i in range(-self.p_r, self.p_r+1):
            j_lim = int(math.sqrt(self.p_r ** 2 - i ** 2))
            for j in range(-j_lim, j_lim + 1):
                if i**2 + j**2 <= self.p_r**2:
                    for k in range(-self.p_d_prime, self.p_d_prime+1):
                        coord = coordinate_3d(i, j, k)
                        coordinates[coord + self.center] = 1
        return coordinates
