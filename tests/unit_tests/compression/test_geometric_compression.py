# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################################################
# Test Geometric Structure based Compression Function #
#######################################################

import pytest
import torch
from tinybig.compression.geometric_compression import (  # Adjust the import according to your file structure
    geometric_compression,
    cuboid_patch_based_geometric_compression,
    cylinder_patch_based_geometric_compression,
    sphere_patch_based_geometric_compression,
    cuboid_max_based_geometric_compression,
    cuboid_min_based_geometric_compression,
    cuboid_mean_based_geometric_compression,
    cylinder_max_based_geometric_compression,
    cylinder_min_based_geometric_compression,
    cylinder_mean_based_geometric_compression,
    sphere_max_based_geometric_compression,
    sphere_min_based_geometric_compression,
    sphere_mean_based_geometric_compression
)
from tinybig.koala.linear_algebra import batch_max, batch_min
from tinybig.koala.statistics import batch_mean

@pytest.fixture
def setup_geometric_compression():
    # Assuming a dummy metric for testing
    def dummy_metric(tensor: torch.Tensor, dim=-1):
        return torch.mean(tensor, dim=dim)  # Just an example metric

    return geometric_compression(metric=dummy_metric, grid_configs={'grid_class': 'tinybig.koala.geometry.grid', 'grid_parameters': {'h': 2, 'w': 2, 'd': 2}}, patch_configs={'patch_class': 'tinybig.koala.geometry.cuboid', 'patch_parameters': {'p_h': 1, 'p_w': 1, 'p_d': 1}})

def test_geometric_compression_initialization(setup_geometric_compression):
    geo_comp = setup_geometric_compression
    assert geo_comp.metric is not None
    assert geo_comp.interdependence.grid is not None
    assert geo_comp.interdependence.patch is not None

def test_geometric_compression_forward(setup_geometric_compression):
    geo_comp = setup_geometric_compression
    x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0]])
    compression = geo_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == geo_comp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_cuboid_max_based_geometric_compression():
    cuboid_max_comp = cuboid_max_based_geometric_compression(h=2, w=2, d=2, p_h=1, p_w=1, p_d=1, metric=batch_max)
    x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0]])
    compression = cuboid_max_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == cuboid_max_comp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_cylinder_mean_based_geometric_compression():
    cylinder_mean_comp = cylinder_mean_based_geometric_compression(h=2, w=2, d=2, p_r=1, p_d=1, metric=batch_mean)
    x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0]])
    compression = cylinder_mean_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == cylinder_mean_comp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_sphere_min_based_geometric_compression():
    sphere_min_comp = sphere_min_based_geometric_compression(h=2, w=2, d=2, p_r=1, metric=batch_min)
    x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0]])
    compression = sphere_min_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == sphere_min_comp.calculate_D(m=x.shape[1])  # Should match calculated D

# Run the tests
if __name__ == "__main__":
    pytest.main()
