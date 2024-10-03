# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#########################################
# Test Grid Structure in koala.geometry #
#########################################

import pytest
import torch
from tinybig.koala.geometry import grid, coordinate_3d, cuboid, cylinder, sphere


@pytest.fixture
def setup_grid():
    # Fixture to set up a grid for testing
    center_coord = coordinate_3d(0, 0, 0)
    return grid(h=3, w=3, d=3, center=center_coord)


class Test_Grid:
    def test_grid_initialization(self, setup_grid):
        g = setup_grid
        assert g.h == 3
        assert g.w == 3
        assert g.d == 3
        assert g.name == "cuboid_geometry"
        assert isinstance(g.center, coordinate_3d)

    def test_grid_generate_coordinates(self, setup_grid):
        g = setup_grid
        coords = g.generate_coordinates()
        # 3x3x3 grid should generate exactly 27 coordinates
        assert len(coords) == 27

        # Ensure the origin and some boundary coordinates are generated correctly
        assert coordinate_3d(0, 0, 0) + g.center in coords
        assert coordinate_3d(2, 2, 2) + g.center in coords

    def test_to_attribute_index(self, setup_grid):
        g = setup_grid
        coord = coordinate_3d(2, 2, 2)
        idx = g.to_attribute_index(coord)
        assert idx == 26  # Based on a 3x3x3 grid

    def test_to_grid_coordinate(self, setup_grid):
        g = setup_grid
        idx = 26
        coord, _ = g.to_grid_coordinate(idx)
        assert coord == coordinate_3d(2, 2, 2)

        # Invalid index should return None
        invalid_idx = 27
        assert g.to_grid_coordinate(invalid_idx)[0] is None

    def test_get_patch_number(self, setup_grid):
        g = setup_grid
        patch_number = g.get_patch_number(cd_h=1, cd_w=1, cd_d=1)
        assert patch_number == 27  # With step size 1 in all dimensions, every grid point is counted

        # With larger step sizes, fewer patches should be counted
        patch_number = g.get_patch_number(cd_h=2, cd_w=2, cd_d=2)
        assert patch_number == 8

    def test_packing(self, setup_grid):
        g = setup_grid
        cuboid_patch = cuboid(p_h=1, p_w=1, p_d=1, center=coordinate_3d(0, 0, 0))
        packed_patches = g.packing(patch=cuboid_patch, cd_h=2, cd_w=2, cd_d=2)

        # Expect a number of packed patches based on the grid dimensions and step sizes
        assert len(packed_patches) == 8
        for center_coord, relative_coords in packed_patches.items():
            assert isinstance(center_coord, coordinate_3d)
            assert isinstance(relative_coords, dict)

    def test_to_aggregation_matrix(self, setup_grid):
        g = setup_grid
        cuboid_patch = cuboid(p_h=1, p_w=1, p_d=1, center=coordinate_3d(0, 0, 0))
        packed_patches = g.packing(patch=cuboid_patch, cd_h=2, cd_w=2, cd_d=2)

        matrix = g.to_aggregation_matrix(packed_patches, n=27, device='cpu')
        assert matrix.size() == torch.Size([27, 27])

    def test_to_padding_matrix(self, setup_grid):
        g = setup_grid
        cuboid_patch = cuboid(p_h=1, p_w=1, p_d=1, center=coordinate_3d(0, 0, 0))
        packed_patches = g.packing(patch=cuboid_patch, cd_h=2, cd_w=2, cd_d=2)

        matrix = g.to_padding_matrix(packed_patches, n=27, device='cpu')
        assert matrix.size() == torch.Size([27, 8 * cuboid_patch.get_volume()])  # 8 packed patches

    def test_to_matrix(self, setup_grid):
        g = setup_grid
        cuboid_patch = cuboid(p_h=1, p_w=1, p_d=0, p_h_prime=0, p_d_prime=0, p_w_prime=0, center=coordinate_3d(0, 0, 0))

        # Test padding matrix mode
        matrix = g.to_matrix(patch=cuboid_patch, cd_h=1, cd_w=1, cd_d=1, interdependence_matrix_mode='padding')
        torch.set_printoptions(threshold=torch.inf, linewidth=10000)
        assert matrix.size() == torch.Size([27, 27 * cuboid_patch.get_volume()])

        # Test aggregation matrix mode
        matrix = g.to_matrix(patch=cuboid_patch, cd_h=1, cd_w=1, cd_d=1, interdependence_matrix_mode='aggregation')
        assert matrix.size() == torch.Size([27, 27])
