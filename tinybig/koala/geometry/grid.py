# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

############################
# Grid Geometric Structure #
############################

import warnings
from typing import Union
import torch

from tinybig.koala.geometry import geometric_space, coordinate_3d, cuboid, cylinder, sphere
from tinybig.koala.linear_algebra import degree_based_normalize_matrix


class grid(geometric_space):
    """
        A class representing a 3D grid in a geometric space, inheriting from the `geometric_space` class.

        The grid is defined by its height, width, depth, and an optional number of universes.
        It provides methods for generating coordinates, converting between grid coordinates and indices,
        and constructing matrices based on grid packing strategies.

        Attributes
        ----------
        h : int
            The height of the grid.
        w : int
            The width of the grid.
        d : int
            The depth of the grid.
        universe_num : int
            The number of universes in the grid.
        center : coordinate_3d
            The central coordinate of the grid.
        name : str
            The name of the grid.
        coordinates : dict
            A dictionary containing all grid coordinates mapped to their presence status.

        Methods
        -------
        generate_coordinates()
            Generates all coordinates in the grid relative to the center.
        to_attribute_index(coord, universe_id)
            Converts a grid coordinate to a unique attribute index.
        to_grid_coordinate(idx)
            Converts an attribute index back to a grid coordinate and universe ID.
        get_patch_num(cd_h, cd_w, cd_d, across_universe)
            Calculates the number of patches based on the center distances and universes.
        get_h()
            Returns the height of the grid.
        get_w()
            Returns the width of the grid.
        get_d()
            Returns the depth of the grid.
        get_grid_size(across_universe)
            Returns the total size of the grid.
        get_grid_shape()
            Returns the dimensions of the grid (height, width, depth).
        get_h_after_packing(cd_h)
            Returns the height after applying a packing strategy.
        get_w_after_packing(cd_w)
            Returns the width after applying a packing strategy.
        get_d_after_packing(cd_d)
            Returns the depth after applying a packing strategy.
        get_grid_shape_after_packing(cd_h, cd_w, cd_d)
            Returns the grid dimensions after packing.
        packing(patch, cd_h, cd_w, cd_d)
            Packs the grid with patches using the specified center distances.
        to_aggregation_matrix(packed_patch, n, across_universe, device)
            Creates an aggregation matrix from packed patches.
        to_padding_matrix(packed_patch, n, across_universe, device)
            Creates a padding matrix from packed patches.
        to_matrix(patch, packing_strategy, cd_h, cd_w, cd_d, interdependence_matrix_mode, normalization, normalization_mode, across_universe, device)
            Constructs a matrix representation of the grid based on the specified parameters.
    """
    def __init__(
        self,
        h: int, w: int,
        d: int = 1, universe_num: int = 1,
        center: coordinate_3d = coordinate_3d(0, 0, 0),
        name: str = 'cuboid_geometry',
        *args, **kwargs
    ):
        """
            Initializes the grid with dimensions, universes, and a central coordinate.

            Parameters
            ----------
            h : int
                The height of the grid.
            w : int
                The width of the grid.
            d : int, optional
                The depth of the grid. Defaults to 1.
            universe_num : int, optional
                The number of universes in the grid. Defaults to 1.
            center : coordinate_3d, optional
                The central coordinate of the grid. Defaults to (0, 0, 0).
            name : str, optional
                The name of the grid. Defaults to 'cuboid_geometry'.
            *args, **kwargs
                Additional arguments for the parent class.

            Raises
            ------
            ValueError
                If any of `h`, `w`, `d`, or `universe_num` is less than or equal to 0.
        """

        if h <= 0 or w <= 0 or d <= 0 or universe_num <= 0:
            raise ValueError("the grid shape configurations needs to be positive...")
        self.h = h
        self.w = w
        self.d = d
        super().__init__(name=name, center=center, universe_num=universe_num, *args, **kwargs)

    def generate_coordinates(self):
        """
            Generates all grid coordinates relative to the center.

            Returns
            -------
            dict
                A dictionary of coordinates where each key is a `coordinate_3d` object
                and the value indicates its presence in the grid.
        """
        coordinates = {}
        for i in range(self.h):
            for j in range(self.w):
                for k in range(self.d):
                    coord = coordinate_3d(i, j, k)
                    coordinates[coord + self.center] = 1
        return coordinates

    def to_attribute_index(self, coord: coordinate_3d, universe_id: int = 0):
        """
            Converts a grid coordinate to a unique attribute index.

            Parameters
            ----------
            coord : coordinate_3d
                The grid coordinate to convert.
            universe_id : int, optional
                The universe ID for the coordinate. Defaults to 0.

            Returns
            -------
            int or None
                The unique attribute index or `None` if the coordinate is not in the grid.

            Raises
            ------
            AssertionError
                If the grid coordinates are not initialized or the input coordinate is None.
        """
        assert self.coordinates is not None and coord is not None
        if coord not in self.coordinates:
            return None
        return coord.d + coord.w * self.d + coord.h * self.d * self.w + universe_id * self.d * self.w * self.h
        #return coord.w + coord.h * self.w + coord.d * self.h * self.w

    def to_grid_coordinate(self, idx: int):
        """
            Converts an attribute index to a grid coordinate and universe ID.

            Parameters
            ----------
            idx : int
                The attribute index to convert.

            Returns
            -------
            tuple
                A tuple containing the `coordinate_3d` object and the universe ID.
                Returns `(None, None)` if the index is invalid or out of bounds.
        """
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
        """
            Calculates the number of patches in the grid.

            Parameters
            ----------
            cd_h : int
                Center distance along the height.
            cd_w : int
                Center distance along the width.
            cd_d : int
                Center distance along the depth.
            across_universe : bool, optional
                If True, includes all universes. Defaults to False.

            Returns
            -------
            int
                The number of patches.

            Raises
            ------
            ValueError
                If any of the center distances is 0.
        """
        if cd_h == 0 or cd_w == 0 or cd_d == 0:
            raise ValueError('patch center distance cannot be zeros...')
        if across_universe:
            return len(range(0, self.h, cd_h)) * len(range(0, self.w, cd_w)) * len(range(0, self.d, cd_d)) * self.universe_num
        else:
            return len(range(0, self.h, cd_h)) * len(range(0, self.w, cd_w)) * len(range(0, self.d, cd_d))

    def get_h(self):
        """
            Returns the height of the grid.

            Returns
            -------
            int
                The height of the grid.
        """
        return self.h

    def get_w(self):
        """
            Returns the width of the grid.

            Returns
            -------
            int
                The width of the grid.
        """
        return self.w

    def get_d(self):
        """
            Returns the depth of the grid.

            Returns
            -------
            int
                The depth of the grid.
        """
        return self.d

    def get_grid_size(self, across_universe: bool = False):
        """
            Returns the total size of the grid.

            Parameters
            ----------
            across_universe : bool, optional
                If True, includes all universes in the size calculation. Defaults to False.

            Returns
            -------
            int
                The total size of the grid.
        """
        return self.get_volume(across_universe=across_universe)

    def get_grid_shape(self):
        """
            Returns the dimensions of the grid.

            Returns
            -------
            tuple
                A tuple (height, width, depth) representing the grid shape.
        """
        return self.get_h(), self.get_w(), self.get_d()

    def get_h_after_packing(self, cd_h: int):
        """
            Returns the height of the grid after packing with a specified center distance.

            Parameters
            ----------
            cd_h : int
                Center distance along the height.

            Returns
            -------
            int
                The packed height.

            Raises
            ------
            ValueError
                If the center distance is 0.
        """
        if cd_h == 0:
            raise ValueError('patch center distance cannot be zeros...')
        return len(range(0, self.h, cd_h))

    def get_w_after_packing(self, cd_w: int):
        """
            Returns the width of the grid after packing with a specified center distance.

            Parameters
            ----------
            cd_w : int
                Center distance along the width.

            Returns
            -------
            int
                The packed width.

            Raises
            ------
            ValueError
                If the center distance is 0.
        """
        if cd_w == 0:
            raise ValueError('patch center distance cannot be zeros...')
        return len(range(0, self.w, cd_w))

    def get_d_after_packing(self, cd_d: int):
        """
            Returns the depth of the grid after packing with a specified center distance.

            Parameters
            ----------
            cd_d : int
                Center distance along the depth.

            Returns
            -------
            int
                The packed depth.

            Raises
            ------
            ValueError
                If the center distance is 0.
        """
        if cd_d == 0:
            raise ValueError('patch center distance cannot be zeros...')
        return len(range(0, self.d, cd_d))

    def get_grid_shape_after_packing(self, cd_h: int, cd_w: int, cd_d: int):
        """
            Returns the grid dimensions after applying a packing strategy.

            Parameters
            ----------
            cd_h : int
                Center distance along the height.
            cd_w : int
                Center distance along the width.
            cd_d : int
                Center distance along the depth.

            Returns
            -------
            tuple
                A tuple (packed_height, packed_width, packed_depth) representing the grid shape after packing.
        """
        return self.get_h_after_packing(cd_h=cd_h), self.get_w_after_packing(cd_w=cd_w), self.get_d_after_packing(cd_d=cd_d)

    def packing(self, patch: cuboid | cylinder | sphere, cd_h: int, cd_w: int, cd_d: int):
        """
            Packs the grid with patches using the specified center distances.

            Parameters
            ----------
            patch : cuboid | cylinder | sphere
                The patch type to use for packing.
            cd_h : int
                Center distance along the height.
            cd_w : int
                Center distance along the width.
            cd_d : int
                Center distance along the depth.

            Returns
            -------
            dict
                A dictionary where keys are patch center coordinates and values are relative coordinates in the patch.

            Raises
            ------
            ValueError
                If any of the center distances is 0 or if the patch center is not at (0, 0, 0).
        """
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
        """
            Creates an aggregation matrix from packed patches.

            Parameters
            ----------
            packed_patch : dict
                A dictionary of packed patches with their relative coordinates.
            n : int
                The total size of the grid.
            across_universe : bool, optional
                If True, includes all universes in the aggregation. Defaults to False.
            device : str, optional
                The device to perform computation on (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args, **kwargs
                Additional arguments for customization.

            Returns
            -------
            torch.Tensor
                A sparse COO tensor representing the aggregation matrix.

            Raises
            ------
            AssertionError
                If row, column, and data lengths do not match.
        """
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
        """
            Creates a padding matrix from packed patches.

            Parameters
            ----------
            packed_patch : dict
                A dictionary of packed patches with their relative coordinates.
            n : int
                The total size of the grid.
            across_universe : bool, optional
                If True, includes all universes in the padding. Defaults to False.
            device : str, optional
                The device to perform computation on (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args, **kwargs
                Additional arguments for customization.

            Returns
            -------
            torch.Tensor
                A sparse COO tensor representing the padding matrix.

            Raises
            ------
            AssertionError
                If row, column, and data lengths do not match.
        """
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
        """
            Constructs a matrix representation of the grid based on the specified parameters.

            Parameters
            ----------
            patch : Union[cuboid, cylinder, sphere]
                The patch type to use for packing.
            packing_strategy : str, optional
                The packing strategy to use. Defaults to None.
            cd_h : int, optional
                Center distance along the height. Defaults to None.
            cd_w : int, optional
                Center distance along the width. Defaults to None.
            cd_d : int, optional
                Center distance along the depth. Defaults to None.
            interdependence_matrix_mode : str, optional
                The mode for the matrix (e.g., 'padding' or 'aggregation'). Defaults to 'padding'.
            normalization : bool, optional
                Whether to normalize the resulting matrix. Defaults to False.
            normalization_mode : str, optional
                The normalization mode (e.g., 'row', 'column', 'row_column'). Defaults to 'row_column'.
            across_universe : bool, optional
                If True, includes all universes in the matrix. Defaults to False.
            device : str, optional
                The device to perform computation on (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args, **kwargs
                Additional arguments for customization.

            Returns
            -------
            torch.Tensor
                A matrix representation of the grid.

            Raises
            ------
            ValueError
                If patch center distances are not positive or if an unknown mode is provided.
        """
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

