# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################################
# Test Cuboid Geometry in koala.geometry #
##########################################

import pytest
from tinybig.koala.geometry import cuboid, coordinate_3d


@pytest.fixture
def setup_cuboid():
    # Fixture to set up a cuboid for testing
    center_coord = coordinate_3d(0, 0, 0)
    return cuboid(p_h=2, p_w=3, p_d=4, center=center_coord)


class Test_Cuboid:
    def test_cuboid_initialization(self, setup_cuboid):
        c = setup_cuboid
        assert c.p_h == 2
        assert c.p_w == 3
        assert c.p_d == 4
        assert c.p_h_prime == 2
        assert c.p_w_prime == 3
        assert c.p_d_prime == 4
        assert c.name == "cuboid_geometry"
        assert isinstance(c.center, coordinate_3d)

    def test_cuboid_shape(self, setup_cuboid):
        c = setup_cuboid
        assert c.shape() == (5, 7, 9)  # height = 2+2+1, width = 3+3+1, depth = 4+4+1

    def test_cuboid_height_width_depth(self, setup_cuboid):
        c = setup_cuboid
        assert c.height() == 5  # 2+2+1
        assert c.width() == 7   # 3+3+1
        assert c.depth() == 9   # 4+4+1

    def test_cuboid_generate_coordinates(self, setup_cuboid):
        c = setup_cuboid
        coords = c.generate_coordinates()
        assert len(coords) == (2*2 + 1) * (2*3 + 1) * (2*4 + 1)  # 5 * 7 * 9 = 315 coordinates
        # Ensure the coordinates include the center and some boundary values
        assert coordinate_3d(0, 0, 0) + c.center in coords
        assert coordinate_3d(2, 3, 4) + c.center in coords
        assert coordinate_3d(-2, -3, -4) + c.center in coords

    def test_cuboid_packing_strategy_parameters(self, setup_cuboid):
        c = setup_cuboid

        # Test default strategy 'complete_square'
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('complete_square')
        assert cd_h == 4  # 2*2
        assert cd_w == 6  # 2*3
        assert cd_d == 8  # 2*4

        # Test 'sparse_square' strategy
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('sparse_square')
        assert cd_h == 6  # 3*2
        assert cd_w == 9  # 3*3
        assert cd_d == 12 # 3*4

        # Test 'dense_square' strategy
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('dense_square')
        assert cd_h == 2  # 2
        assert cd_w == 3  # 3
        assert cd_d == 4  # 4

        # Test 'densest_packing' strategy
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('densest_packing')
        assert cd_h == 1
        assert cd_w == 1
        assert cd_d == 1

        # Test unknown strategy, should fallback to 'densest_packing'
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('unknown_strategy')
        assert cd_h == 1
        assert cd_w == 1
        assert cd_d == 1

    def test_cuboid_get_packing_strategies(self):
        strategies = cuboid.get_packing_strategies()
        assert strategies == ['sparse_square', 'complete_square', 'dense_square', 'densest_packing']
