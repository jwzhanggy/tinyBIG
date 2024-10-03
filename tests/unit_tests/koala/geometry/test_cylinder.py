# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

############################################
# Test Cylinder Geometry in koala.geometry #
############################################

import pytest
from tinybig.koala.geometry import cylinder, coordinate_3d
import math


@pytest.fixture
def setup_cylinder():
    # Fixture to set up a cylinder for testing
    center_coord = coordinate_3d(0, 0, 0)
    return cylinder(p_r=5, p_d=7, center=center_coord)


class Test_Cylinder:
    def test_cylinder_initialization(self, setup_cylinder):
        c = setup_cylinder
        assert c.p_r == 5
        assert c.p_d == 7
        assert c.p_d_prime == 7
        assert c.name == "cylinder_geometry"
        assert isinstance(c.center, coordinate_3d)

    def test_cylinder_radius(self, setup_cylinder):
        c = setup_cylinder
        assert c.radius() == 5

    def test_cylinder_depth(self, setup_cylinder):
        c = setup_cylinder
        assert c.depth() == 15  # 7 + 7 + 1

    def test_cylinder_generate_coordinates(self, setup_cylinder):
        c = setup_cylinder
        coords = c.generate_coordinates()
        # Roughly calculate the number of coordinates within the cylinder's radius and depth
        num_coordinates = sum(1 for i in range(-5, 6) for j in range(-int(math.sqrt(5 ** 2 - i ** 2)), int(math.sqrt(5 ** 2 - i ** 2)) + 1) for k in range(-7, 8))
        assert len(coords) == num_coordinates

        # Ensure some specific coordinates are generated correctly
        assert coordinate_3d(0, 0, 0) + c.center in coords
        assert coordinate_3d(5, 0, 7) + c.center in coords
        assert coordinate_3d(-5, 0, -7) + c.center in coords

    def test_cylinder_packing_strategy_parameters(self, setup_cylinder):
        c = setup_cylinder

        # Test default strategy 'complete_square'
        cd_h, cd_w, cd_d = c.packing_strategy_parameters(packing_strategy='complete_square')
        assert math.isclose(cd_h, math.sqrt(2) * 5, rel_tol=1e-01)
        assert math.isclose(cd_w, math.sqrt(2) * 5, rel_tol=1e-01)
        assert cd_d == 2 * 7

        # Test 'sparse_square' strategy
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('sparse_square')
        assert cd_h == 2 * 5
        assert cd_w == 2 * 5
        assert cd_d == 2 * 7

        # Test 'sparse_hexagonal' strategy
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('sparse_hexagonal')
        assert math.isclose(cd_h, math.sqrt(3) * 5, rel_tol=1e-01)
        assert cd_w == 2 * 5
        assert cd_d == 2 * 7

        # Test 'complete_hexagonal' strategy
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('complete_hexagonal')
        assert math.isclose(cd_h, 1.5 * 5, rel_tol=1e-01)
        assert math.isclose(cd_w, math.sqrt(3) * 5, rel_tol=1e-01)
        assert cd_d == 2 * 7

        # Test 'dense_hexagonal' strategy
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('dense_hexagonal')
        assert math.isclose(cd_h, 0.5 * math.sqrt(6) * 5, rel_tol=1e-01)
        assert math.isclose(cd_w, math.sqrt(2) * 5, rel_tol=1e-01)
        assert cd_d == 7

        # Test 'denser_hexagonal' strategy
        cd_h, cd_w, cd_d = c.packing_strategy_parameters('denser_hexagonal')
        assert math.isclose(cd_h, 0.5 * math.sqrt(3) * 5, rel_tol=1e-01)
        assert cd_w == 5
        assert cd_d == 7

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

    def test_cylinder_get_packing_strategies(self):
        strategies = cylinder.get_packing_strategies()
        assert strategies == [
            'sparse_square', 'complete_square', 'sparse_hexagonal',
            'complete_hexagonal', 'dense_hexagonal', 'denser_hexagonal', 'densest_packing'
        ]
