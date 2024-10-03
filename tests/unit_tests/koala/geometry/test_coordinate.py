# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################
# Test Coordinate in koala.geometry #
#####################################

import pytest
from tinybig.koala.geometry.coordinate import coordinate, coordinate_3d, coordinate_2d


class Test_Coordinate:
    @pytest.mark.parametrize("coords1, coords2, expected", [
        ((1,), (4,), (5,)),
        ((1, 2), (4, 5), (5, 7)),
        ((1, 2, 3), (4, 5, 6), (5, 7, 9)),
        ((1, 2, 3, 4), (4, 5, 6, 7), (5, 7, 9, 11)),
        ((-1, -2, -3), (1, 2, 3), (0, 0, 0)),
    ])

    def test_coordinate_add(self, coords1, coords2, expected):
        coord1 = coordinate(coords1)
        coord2 = coordinate(coords2)
        result = coord1 + coord2
        assert result.coords == expected

    @pytest.mark.parametrize("coords1, coords2, expected", [
        ((5,), (4,), (1,)),
        ((5, 7), (4, 5), (1, 2)),
        ((5, 7, 9), (4, 5, 6), (1, 2, 3)),
        ((5, 7, 9, 11), (4, 5, 6, 7), (1, 2, 3, 4)),
        ((1, 1, 1), (-1, -2, -3), (2, 3, 4)),
    ])
    def test_coordinate_sub(self, coords1, coords2, expected):
        coord1 = coordinate(coords1)
        coord2 = coordinate(coords2)
        result = coord1 - coord2
        assert result.coords == expected

    @pytest.mark.parametrize("coords1, coords2, expected", [
        ((1,), (1,), True),
        ((1, 2), (1, 2), True),
        ((1, 2, 3), (1, 2, 3), True),
        ((1, 2, 3, 4), (1, 2, 3, 4), True),
        ((1, 2, 3), (4, 5, 6), False),
    ])
    def test_coordinate_equality(self, coords1, coords2, expected):
        coord1 = coordinate(coords1)
        coord2 = coordinate(coords2)
        assert (coord1 == coord2) == expected

    @pytest.mark.parametrize("coords1, coords2, expected", [
        ((1,), (4,), True),
        ((1, 2), (4, 5), True),
        ((1, 2, 3), (4, 5, 6), True),
        ((1, 2, 3, 4), (4, 5, 6, 7), True),
        ((4, 5, 6), (1, 2, 3), False),
    ])
    def test_coordinate_comparison(self, coords1, coords2, expected):
        coord1 = coordinate(coords1)
        coord2 = coordinate(coords2)
        assert (coord1 < coord2) == expected

    def test_coordinate_repr(self):
        coord = coordinate((1, 2, 3))
        assert repr(coord) == "coordinate(1, 2, 3)"

    @pytest.mark.parametrize("coords1, coords2, expected_kernel", [
        ((1,), (4,), 'euclidean_distance'),
        ((1, 2), (4, 5), 'manhattan_distance'),
        ((1, 2, 3), (4, 5, 6), 'euclidean_distance'),
        ((-1, -2, -3), (1, 2, 3), 'manhattan_distance'),
    ])
    def test_coordinate_distance(self, coords1, coords2, expected_kernel):
        coord1 = coordinate(coords1)
        coord2 = coordinate(coords2)
        result = coord1.distance_to(coord2, kernel_name=expected_kernel)
        print(result, expected_kernel)
        assert isinstance(result, float)


class Test_Coordinate_3D:

    @pytest.mark.parametrize("h, w, d, expected_coords", [
        (1, 2, 3, (1, 2, 3)),
        (0, 0, 0, (0, 0, 0)),
        (-1, -2, -3, (-1, -2, -3)),
    ])
    def test_coordinate_3d_initialization(self, h, w, d, expected_coords):
        coord_3d = coordinate_3d(h, w, d)
        assert coord_3d.coords == expected_coords

    def test_coordinate_3d_setters_getters(self):
        coord_3d = coordinate_3d(1, 2, 3)

        assert coord_3d.h == 1
        assert coord_3d.w == 2
        assert coord_3d.d == 3

        coord_3d.h = 10
        coord_3d.w = 20
        coord_3d.d = 30

        assert coord_3d.h == 10
        assert coord_3d.w == 20
        assert coord_3d.d == 30

    def test_coordinate_3d_aliases(self):
        coord_3d = coordinate_3d(1, 2, 3)

        assert coord_3d.x == 1
        assert coord_3d.y == 2
        assert coord_3d.z == 3

        coord_3d.x = 10
        coord_3d.y = 20
        coord_3d.z = 30

        assert coord_3d.x == 10
        assert coord_3d.y == 20
        assert coord_3d.z == 30


class Test_Coordinate_2D:

    @pytest.mark.parametrize("h, w, expected_coords", [
        (1, 2, (1, 2)),
        (0, 0, (0, 0)),
        (-1, -2, (-1, -2)),
    ])
    def test_coordinate_2d_initialization(self, h, w, expected_coords):
        coord_2d = coordinate_2d(h, w)
        assert coord_2d.coords == expected_coords

    def test_coordinate_2d_setters_getters(self):
        coord_3d = coordinate_2d(1, 2)

        assert coord_3d.h == 1
        assert coord_3d.w == 2

        coord_3d.h = 10
        coord_3d.w = 20

        assert coord_3d.h == 10
        assert coord_3d.w == 20

    def test_coordinate_2d_aliases(self):
        coord_3d = coordinate_2d(1, 2)

        assert coord_3d.x == 1
        assert coord_3d.y == 2

        coord_3d.x = 10
        coord_3d.y = 20

        assert coord_3d.x == 10
        assert coord_3d.y == 20