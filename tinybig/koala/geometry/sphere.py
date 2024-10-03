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

    def __init__(
        self,
        p_r: int,
        center: coordinate_3d = coordinate_3d(0, 0, 0),
        name: str = 'sphere_geometry',
        *args, **kwargs
    ):
        self.p_r = p_r
        super().__init__(center=center, name=name, *args, **kwargs)

    def radius(self):
        return self.p_r

    @staticmethod
    def get_packing_strategies():
        return [
            'sparse_simple_cubic', 'complete_simple_cubic', 'dense_simple_cubic',
            'sparse_face_centered_cubic', 'complete_face_centered_cubic', 'dense_face_centered_cubic',
            'sparse_hexagonal', 'complete_hexagonal', 'dense_hexagonal',
            'densest_packing',
        ]

    def packing_strategy_parameters(self, packing_strategy: str = 'complete_square', *args, **kwargs):

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

        return int(cd_h), int(cd_w), int(cd_d)

    def generate_coordinates(self, *args, **kwargs):
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
