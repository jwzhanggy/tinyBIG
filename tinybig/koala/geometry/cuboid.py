# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################
# Cuboid Geometric Structure #
##############################

import warnings

from tinybig.koala.geometry import geometric_space, coordinate_3d


class cuboid(geometric_space):
    def __init__(
        self,
        p_h: int, p_w: int, p_d: int = 0,
        p_h_prime: int = None, p_w_prime: int = None, p_d_prime: int = None,
        center: coordinate_3d = coordinate_3d(0, 0, 0),
        name: str = 'cuboid_geometry',
        *args, **kwargs
    ):
        self.p_h = p_h
        self.p_w = p_w
        self.p_d = p_d
        self.p_h_prime = p_h_prime if p_h_prime is not None else p_h
        self.p_w_prime = p_w_prime if p_w_prime is not None else p_w
        self.p_d_prime = p_d_prime if p_d_prime is not None else p_d
        super().__init__(center=center, name=name, *args, **kwargs)

    def shape(self):
        return self.height(), self.width(), self.depth()

    def height(self):
        return self.p_h + self.p_h_prime + 1

    def width(self):
        return self.p_w + self.p_w_prime + 1

    def depth(self):
        return self.p_d + self.p_d_prime + 1

    @staticmethod
    def get_packing_strategies():
        return ['sparse_square', 'complete_square', 'dense_square', 'densest_packing']

    def packing_strategy_parameters(self, packing_strategy: str = 'complete_square', *args, **kwargs):

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
        coordinates = {}
        for i in range(-self.p_h, self.p_h_prime+1):
            for j in range(-self.p_w, self.p_w_prime+1):
                for k in range(-self.p_d, self.p_d_prime+1):
                    coord = coordinate_3d(i, j, k)
                    coordinates[coord + self.center] = 1
        return coordinates
