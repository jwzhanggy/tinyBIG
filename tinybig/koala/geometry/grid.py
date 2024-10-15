# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

############################
# Grid Geometric Structure #
############################

import warnings
from typing import Union
import torch
import time

from tinybig.koala.geometry import geometric_space, coordinate_3d, cuboid, cylinder, sphere
from tinybig.koala.linear_algebra import degree_based_normalize_matrix


class grid(geometric_space):
    def __init__(
        self,
        h: int, w: int,
        d: int = 1, universe_num: int = 1,
        center: coordinate_3d = coordinate_3d(0, 0, 0),
        name: str = 'cuboid_geometry',
        *args, **kwargs
    ):
        if h <= 0 or w <= 0 or d <= 0 or universe_num <= 0:
            raise ValueError("the grid shape configurations needs to be positive...")
        self.h = h
        self.w = w
        self.d = d
        super().__init__(name=name, center=center, universe_num=universe_num, *args, **kwargs)

    def generate_coordinates(self):
        coordinates = {}
        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.d):
                    coord = coordinate_3d(i, j, k)
                    coordinates[coord + self.center] = 1
        return coordinates

    def to_attribute_index(self, coord: coordinate_3d, universe_id: int = 0):
        assert self.coordinates is not None and coord is not None
        if coord not in self.coordinates:
            return None
        return coord.d + coord.w * self.d + coord.h * self.d * self.w + universe_id * self.d * self.w * self.h
        #return coord.w + coord.h * self.w + coord.d * self.h * self.w

    def to_grid_coordinate(self, idx: int):
        universe_id = int(idx/(self.h * self.w * self.d))
        h = int((idx % (self.h * self.w * self.d))/(self.d * self.w))
        w = int((idx % (self.d * self.w))/self.d)
        d = int(idx % self.d)

        #d = int(idx/(self.w * self.h))
        #h = int((idx % (self.w * self.h))/self.w)
        #w = idx % self.w
        if coordinate_3d(h, w, d) in self.coordinates and 0 <= universe_id < self.universe_num:
            return coordinate_3d(h, w, d), universe_id
        else:
            return None, None

    def get_patch_num(self, cd_h: int, cd_w: int, cd_d: int, across_universe: bool = False):
        if cd_h == 0 or cd_w == 0 or cd_d == 0:
            raise ValueError('patch center distance cannot be zeros...')
        if across_universe:
            return len(range(0, self.h, cd_h)) * len(range(0, self.w, cd_w)) * len(range(0, self.d, cd_d)) * self.universe_num
        else:
            return len(range(0, self.h, cd_h)) * len(range(0, self.w, cd_w)) * len(range(0, self.d, cd_d))

    def get_h(self):
        return self.h

    def get_w(self):
        return self.w

    def get_d(self):
        return self.d

    def get_grid_size(self, across_universe: bool = False):
        return self.get_volume(across_universe=across_universe)

    def get_grid_shape(self):
        return self.get_h(), self.get_w(), self.get_d()

    def get_h_after_packing(self, cd_h: int):
        if cd_h == 0:
            raise ValueError('patch center distance cannot be zeros...')
        return len(range(0, self.h, cd_h))

    def get_w_after_packing(self, cd_w: int):
        if cd_w == 0:
            raise ValueError('patch center distance cannot be zeros...')
        return len(range(0, self.w, cd_w))

    def get_d_after_packing(self, cd_d: int):
        if cd_d == 0:
            raise ValueError('patch center distance cannot be zeros...')
        return len(range(0, self.d, cd_d))

    def get_grid_shape_after_packing(self, cd_h: int, cd_w: int, cd_d: int):
        return self.get_h_after_packing(cd_h=cd_h), self.get_w_after_packing(cd_w=cd_w), self.get_d_after_packing(cd_d=cd_d)

    def packing(self, patch: cuboid | cylinder | sphere, cd_h: int, cd_w: int, cd_d: int):
        if cd_h == 0 or cd_w == 0 or cd_d == 0:
            raise ValueError('patch center distance cannot be zeros...')

        if patch.center is not coordinate_3d(0, 0, 0):
            patch.update_center(new_center=coordinate_3d(0, 0, 0))

        packed_patch = {}
        for i in range(0, self.h, cd_h):
            for j in range(0, self.w, cd_w):
                for k in range(0, self.d, cd_d):
                    center_coord = coordinate_3d(i, j, k)
                    packed_patch[center_coord] = patch.get_relative_coordinates(center=center_coord)
        return packed_patch

    def to_aggregation_matrix(self, packed_patch: dict, n: int, across_universe: bool = False, device: str = 'cpu', *args, **kwargs):
        universe_num = self.universe_num if across_universe else 1
        rows, columns, data = [], [], []

        for patch_center, contexts in packed_patch.items():
            for universe_id in range(universe_num):
                for coord, value in contexts.items():
                    column_idx = self.to_attribute_index(coord=patch_center, universe_id=universe_id)
                    row_idx = self.to_attribute_index(coord=coord, universe_id=universe_id)
                    # check row, column index validity
                    if coord in self.coordinates and row_idx is not None and 0 <= row_idx < n and column_idx is not None and 0 <= column_idx < n:
                        rows.append(row_idx)
                        columns.append(column_idx)
                        data.append(1.0)

        assert len(rows) == len(columns) == len(data)
        if device == 'mps':
            mx = torch.zeros((n, n), device=device)
            mx[torch.tensor(rows, device=device), torch.tensor(columns, device=device)] = torch.tensor(data, device=device)
        else:
            mx = torch.sparse_coo_tensor(torch.tensor([rows, columns]), values=torch.tensor(data), size=(n, n), device=device)
        return mx

    def to_padding_matrix(self, packed_patch: dict, n: int, across_universe: bool = False, device: str = 'cpu', *args, **kwargs):
        universe_num = self.universe_num if across_universe else 1
        all_rows, all_columns, all_data = [], [], []

        for _, contexts in packed_patch.items():
            for universe_id in range(universe_num):
                rows, column, data = [], [], []
                for column_idx, coords in enumerate(contexts.keys()):
                    row_idx = self.to_attribute_index(coords, universe_id=universe_id)
                    # check row index validity
                    if coords in self.coordinates and row_idx is not None and 0 <= row_idx < n:
                        value = 1.0
                    else:
                        row_idx = 0
                        value = 0.0
                    rows.append(row_idx)
                    column.append(column_idx + len(all_columns))
                    data.append(value)
                all_rows.extend(rows)
                all_columns.extend(column)
                all_data.extend(data)

        assert len(all_rows) == len(all_columns) == len(all_data)

        if device == 'mps':
            mx = torch.zeros((n, len(all_columns)), device=device)
            mx[torch.tensor(all_rows, device=device), torch.tensor(all_columns, device=device)] = torch.tensor(all_data, device=device)
        else:
            mx = torch.sparse_coo_tensor(torch.tensor([all_rows, all_columns]), values=torch.tensor(all_data), size=(n, len(all_columns)), device=device)
        return mx

    def to_matrix(
        self,
        patch: Union[cuboid, cylinder, sphere],
        packing_strategy: str = None,
        cd_h: int = None,
        cd_w: int = None,
        cd_d: int = None,
        interdependence_matrix_mode: str = 'padding',
        normalization: bool = False,
        normalization_mode: str = 'row_column',
        across_universe: bool = False,
        device: str = 'cpu',
        *args, **kwargs
    ):
        cd_h, cd_w, cd_d = cd_h, cd_w, cd_d if (cd_h is not None and cd_w is not None and cd_d is not None) else patch.packing_strategy_parameters(packing_strategy=packing_strategy)
        if cd_h <= 0 or cd_w <= 0 or cd_d <= 0:
            raise ValueError('patch center distance should be positive...')

        packed_patch = self.packing(patch=patch, cd_h=cd_h, cd_w=cd_w, cd_d=cd_d)

        if interdependence_matrix_mode == 'padding':
            adj = self.to_padding_matrix(packed_patch=packed_patch, n=self.get_volume(across_universe=across_universe), across_universe=across_universe, device=device)
        elif interdependence_matrix_mode == 'aggregation':
            adj = self.to_aggregation_matrix(packed_patch=packed_patch, n=self.get_volume(across_universe=across_universe), across_universe=across_universe,  device=device)
        else:
            warnings.warn("Unknown mode '{}', will use the default padding mode...".format(interdependence_matrix_mode))
            adj = self.to_padding_matrix(packed_patch=packed_patch, n=self.get_volume(across_universe=across_universe), across_universe=across_universe, device=device)

        if normalization:
            adj = degree_based_normalize_matrix(mx=adj, mode=normalization_mode)

        return adj

