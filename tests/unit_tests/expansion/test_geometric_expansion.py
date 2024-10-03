# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################################
# Test Geometric Structure based Expansion Function #
#####################################################

import pytest
import torch
from tinybig.expansion.geometric_expansion import (  # Adjust the import according to your file structure
    geometric_expansion,
    cuboid_patch_based_geometric_expansion,
    cylinder_patch_based_geometric_expansion,
    sphere_patch_based_geometric_expansion,
)

@pytest.fixture
def setup_geometric_expansion():
    # Assuming a dummy metric for testing
    def dummy_metric(tensor: torch.Tensor):
        return torch.mean(tensor, dim=-1)  # Example metric

    return geometric_expansion(
        name='test_geometric_expansion',
        metric=dummy_metric,
        grid_configs={'grid_class': 'tinybig.koala.geometry.grid', 'grid_parameters': {'h': 2, 'w': 2, 'd': 2}},
        patch_configs={'patch_class': 'tinybig.koala.geometry.cuboid', 'patch_parameters': {'p_h': 1, 'p_w': 1, 'p_d': 1}}
    )

def test_geometric_expansion_initialization(setup_geometric_expansion):
    geo_exp = setup_geometric_expansion
    assert geo_exp.interdependence is not None

def test_geometric_expansion_calculate_D(setup_geometric_expansion):
    geo_exp = setup_geometric_expansion
    # Assuming the grid size is set to match the expected m
    grid_size = geo_exp.interdependence.get_grid_size()
    assert geo_exp.calculate_D(m=grid_size) > 0  # Ensure it returns a positive value

def test_geometric_expansion_forward(setup_geometric_expansion):
    geo_exp = setup_geometric_expansion
    x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0]])
    expansion = geo_exp.forward(x)
    assert expansion.shape[0] == 2  # Should match batch size
    assert expansion.shape[1] == geo_exp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_cuboid_patch_based_geometric_expansion():
    cuboid_exp = cuboid_patch_based_geometric_expansion(h=2, w=2, d=2, p_h=1, p_w=1, p_d=1, name='test_cuboid')
    x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0]])
    expansion = cuboid_exp.forward(x)
    assert expansion.shape[0] == 2  # Should match batch size
    assert expansion.shape[1] == cuboid_exp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_cylinder_patch_based_geometric_expansion():
    cylinder_exp = cylinder_patch_based_geometric_expansion(h=2, w=2, d=2, p_r=1, p_d=1, name='test_cylinder')
    x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0]])
    expansion = cylinder_exp.forward(x)
    assert expansion.shape[0] == 2  # Should match batch size
    assert expansion.shape[1] == cylinder_exp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_sphere_patch_based_geometric_expansion():
    sphere_exp = sphere_patch_based_geometric_expansion(h=2, w=2, d=2, p_r=1, name='test_sphere')
    x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0]])
    expansion = sphere_exp.forward(x)
    assert expansion.shape[0] == 2  # Should match batch size
    assert expansion.shape[1] == sphere_exp.calculate_D(m=x.shape[1])  # Should match calculated D

# Run the tests
if __name__ == "__main__":
    pytest.main()
