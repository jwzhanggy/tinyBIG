# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###################################
# Test Geometry in koala.geometry #
###################################

import pytest
from tinybig.koala.geometry import geometric_space, coordinate


class concrete_geometry(geometric_space):
    def generate_coordinates(self, *args, **kwargs):
        return {
            self.center: 1,
            self.center + coordinate((1, 1, 1)): 1,
            self.center + coordinate((-1, -1, -1)): 1
        }


@pytest.fixture
def setup_geometry():
    # Fixture to set up a basic geometry for testing
    center_coord = coordinate((0, 0, 0))
    return concrete_geometry(center=center_coord)


class Test_Geometry:

    def test_geometry_initialization(self, setup_geometry):
        geometry = setup_geometry
        assert geometry.name == "base_geometry"
        assert geometry.device == "cpu"
        assert isinstance(geometry.center, coordinate)
        assert geometry.get_volume() == 3  # The concrete implementation returns 3 coordinates

    def test_geometry_get_coordinates(self, setup_geometry):
        geometry = setup_geometry
        coords = geometry.get_coordinates()
        assert len(coords) == 3
        assert all(isinstance(coord, coordinate) for coord in coords)

    def test_geometry_update_center(self, setup_geometry):
        geometry = setup_geometry
        new_center = coordinate((1, 1, 1))
        geometry.update_center(new_center)

        # Verify the center has been updated
        assert geometry.center == new_center

        # Verify the relative coordinates are updated
        relative_coords = geometry.get_coordinates()
        assert len(relative_coords) == 3

        expected_relative_coords = {
            new_center: 1,
            new_center + coordinate((1, 1, 1)): 1,
            new_center + coordinate((-1, -1, -1)): 1
        }
        assert all(coord in relative_coords for coord in expected_relative_coords)

    def test_geometry_get_volume(self, setup_geometry):
        geometry = setup_geometry
        assert geometry.get_volume() == 3  # As defined in the concrete_geometry class


import pytest
import torch
from tinybig.koala.geometry import coordinate
from tinybig.koala.linear_algebra import euclidean_distance
from tinybig.koala.geometry import geometric_space

# Dummy subclass of geometric_space to implement abstract method
class test_geometric_space(geometric_space):
    def generate_coordinates(self):
        # For simplicity, we create a basic set of coordinates
        return [coordinate([0, 0]), coordinate([1, 1]), coordinate([2, 2])]

@pytest.fixture
def geom_space():
    center = coordinate([0, 0])
    return test_geometric_space(center=center, universe_num=2)

def test_valid_distance_within_universe(geom_space):
    coord1 = coordinate([0, 0])
    coord2 = coordinate([1, 1])
    distance = geom_space.calculate_distance(coord1, 0, coord2, 0)
    expected_distance = euclidean_distance(torch.Tensor([0, 0]), torch.Tensor([1, 1]))
    assert torch.isclose(distance, expected_distance), f"Expected {expected_distance}, but got {distance}"

def test_invalid_coordinates(geom_space):
    coord1 = coordinate([0, 0])
    coord_invalid = coordinate([10, 10])  # Not in generated coordinates
    with pytest.raises(ValueError, match="coordinates do not exist in the current geometric space"):
        geom_space.calculate_distance(coord1, 0, coord_invalid, 0)

def test_invalid_universe_id(geom_space):
    coord1 = coordinate([0, 0])
    coord2 = coordinate([1, 1])
    with pytest.raises(ValueError, match="universe ids do not exist in the current geometric space"):
        geom_space.calculate_distance(coord1, 0, coord2, 2)  # Invalid universe id

def test_cross_universe_distance(geom_space):
    coord1 = coordinate([0, 0])
    coord2 = coordinate([1, 1])
    with pytest.raises(ValueError, match="the current distance metric can only calculate distance within the same universe space"):
        geom_space.calculate_distance(coord1, 0, coord2, 1)  # Cross-universe distance

def test_missing_distance_metric(geom_space):
    geom_space.distance_metric = None  # Set distance metric to None
    coord1 = coordinate([0, 0])
    coord2 = coordinate([1, 1])
    with pytest.raises(ValueError, match="distance_metric must be defined"):
        geom_space.calculate_distance(coord1, 0, coord2, 0)
