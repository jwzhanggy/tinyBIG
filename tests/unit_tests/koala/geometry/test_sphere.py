# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################################
# Test Sphere Geometry in koala.geometry #
##########################################

import pytest
from tinybig.koala.geometry import sphere, coordinate_3d
import math


@pytest.fixture
def setup_sphere():
    # Fixture to set up a sphere for testing
    center_coord = coordinate_3d(0, 0, 0)
    return sphere(p_r=5, center=center_coord)


class Test_Sphere:

    def test_sphere_initialization(self, setup_sphere):
        s = setup_sphere
        assert s.p_r == 5
        assert s.name == "sphere_geometry"
        assert isinstance(s.center, coordinate_3d)

    def test_sphere_radius(self, setup_sphere):
        s = setup_sphere
        assert s.radius() == 5

    def test_sphere_generate_coordinates(self, setup_sphere):
        s = setup_sphere
        coords = s.generate_coordinates()
        # Roughly calculate the number of coordinates within the sphere's radius
        num_coordinates = sum(
            1 for i in range(-5, 6)
            for j in range(-int(math.sqrt(5 ** 2 - i ** 2)), int(math.sqrt(5 ** 2 - i ** 2)) + 1)
            for k in range(-int(math.sqrt(5 ** 2 - i ** 2 - j ** 2)), int(math.sqrt(5 ** 2 - i ** 2 - j ** 2)) + 1)
            if i**2 + j**2 + k**2 <= 5**2
        )
        assert len(coords) == num_coordinates

        # Ensure some specific coordinates are generated correctly
        assert coordinate_3d(0, 0, 0) + s.center in coords
        assert coordinate_3d(5, 0, 0) + s.center in coords
        assert coordinate_3d(-5, 0, 0) + s.center in coords
        assert coordinate_3d(0, 5, 0) + s.center in coords
        assert coordinate_3d(0, 0, 5) + s.center in coords

    def test_sphere_packing_strategy_parameters(self, setup_sphere):
        s = setup_sphere

        # Test default strategy 'complete_simple_cubic'
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('complete_simple_cubic')
        assert math.isclose(cd_h, 0.6667 * math.sqrt(3) * 5, rel_tol=0.9)
        assert math.isclose(cd_w, 0.6667 * math.sqrt(3) * 5, rel_tol=0.9)
        assert math.isclose(cd_d, 0.6667 * math.sqrt(3) * 5, rel_tol=0.9)

        # Test 'sparse_simple_cubic' strategy
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('sparse_simple_cubic')
        assert cd_h == 2 * 5
        assert cd_w == 2 * 5
        assert cd_d == 2 * 5

        # Test 'dense_simple_cubic' strategy
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('dense_simple_cubic')
        assert cd_h == 5
        assert cd_w == 5
        assert cd_d == 5

        # Test 'sparse_face_centered_cubic' strategy
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('sparse_face_centered_cubic')
        assert cd_h == 2 * 5
        assert math.isclose(cd_w, math.sqrt(3) * 5, rel_tol=0.9)
        assert math.isclose(cd_d, 0.6667 * math.sqrt(6) * 5, rel_tol=0.9)

        # Test 'complete_face_centered_cubic' strategy
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('complete_face_centered_cubic')
        assert math.isclose(cd_h, math.sqrt(2) * 5, rel_tol=0.9)
        assert math.isclose(cd_w, 0.6667 * math.sqrt(6) * 5, rel_tol=0.9)
        assert math.isclose(cd_d, 1.3333 * 5, rel_tol=0.9)

        # Test 'dense_face_centered_cubic' strategy
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('dense_face_centered_cubic')
        assert cd_h == 5
        assert cd_w == 5
        assert cd_d == 5

        # Test 'sparse_hexagonal' strategy
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('sparse_hexagonal')
        assert cd_h == 2 * 5
        assert cd_w == 2 * 5
        assert math.isclose(cd_d, math.sqrt(6) * 5, rel_tol=0.9)

        # Test 'complete_hexagonal' strategy
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('complete_hexagonal')
        assert math.isclose(cd_h, 0.6667 * math.sqrt(3) * 5, rel_tol=0.9)
        assert math.isclose(cd_w, 0.6667 * math.sqrt(3) * 5, rel_tol=0.9)
        assert cd_d == 5

        # Test 'dense_hexagonal' strategy
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('dense_hexagonal')
        assert cd_h == 5
        assert cd_w == 5
        assert cd_d == 5

        # Test 'densest_packing' strategy
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('densest_packing')
        assert cd_h == 1
        assert cd_w == 1
        assert cd_d == 1

        # Test unknown strategy, should fallback to 'densest_packing'
        cd_h, cd_w, cd_d = s.packing_strategy_parameters('unknown_strategy')
        assert cd_h == 1
        assert cd_w == 1
        assert cd_d == 1

    def test_sphere_get_packing_strategies(self):
        strategies = sphere.get_packing_strategies()
        assert strategies == [
            'sparse_simple_cubic', 'complete_simple_cubic', 'dense_simple_cubic',
            'sparse_face_centered_cubic', 'complete_face_centered_cubic', 'dense_face_centered_cubic',
            'sparse_hexagonal', 'complete_hexagonal', 'dense_hexagonal',
            'densest_packing'
        ]
