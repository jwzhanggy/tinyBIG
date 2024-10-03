# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Test Geometric Interdependence #
##################################

import pytest
import torch
from tinybig.interdependence.geometric_interdependence import geometric_interdependence
from tinybig.koala.geometry import grid as grid_structure
from tinybig.koala.geometry import cuboid, sphere, cylinder

device = 'mps'

# try multi-universe 3d space first
@pytest.fixture
def sample_grid_and_patch():
    grid = grid_structure(h=10, w=8, d=5, universe_num=3)
    b, m = 5, grid.get_volume()*3
    x = torch.tensor([list(range(m))] * 5, dtype=torch.float32, device=device)
    patch1 = cuboid(p_h=1, p_w=1, p_d=1)
    patch2 = sphere(p_r=2)
    patch3 = cylinder(p_r=2, p_d=1)
    return x, grid, patch1, patch2, patch3


@pytest.mark.parametrize("patch_type", ['cuboid', 'sphere', 'cylinder'])
def test_geometric_interdependence_initialization(sample_grid_and_patch, patch_type):
    """
    Test the initialization of geometric_interdependence class with different patches.
    """
    x, grid, patch1, patch2, patch3 = sample_grid_and_patch

    # Select patch based on patch_type
    if patch_type == 'cuboid':
        patch = patch1
    elif patch_type == 'sphere':
        patch = patch2
    elif patch_type == 'cylinder':
        patch = patch3
    else:
        patch = patch1

    b, m = x.shape

    # Test successful initialization
    geo_interdep = geometric_interdependence(
        b=b, m=m, interdependence_type='attribute',
        grid=grid, patch=patch,
        interdependence_matrix_mode='padding', normalization=True,
        device=device
    )

    assert geo_interdep.grid == grid
    assert geo_interdep.patch == patch
    assert geo_interdep.interdependence_type == 'attribute'
    assert geo_interdep.packing_strategy == 'densest_packing'

    # Test invalid grid initialization
    with pytest.raises(ValueError):
        geometric_interdependence(
            b=b, m=m, interdependence_type='attribute',
            grid=None, patch=patch
        )

    # Test invalid patch initialization
    with pytest.raises(ValueError):
        geometric_interdependence(
            b=b, m=m, interdependence_type='attribute',
            grid=grid, patch=None
        )


@pytest.mark.parametrize("patch_type", ['cuboid'])
def test_geometric_interdependence_initialization(sample_grid_and_patch, patch_type):
    """
    Test the initialization of geometric_interdependence class with different patches.
    """
    x, grid, patch1, patch2, patch3 = sample_grid_and_patch

    # Select patch based on patch_type
    if patch_type == 'cuboid':
        patch = patch1
    elif patch_type == 'sphere':
        patch = patch2
    elif patch_type == 'cylinder':
        patch = patch3
    else:
        patch = patch1

    b, m = x.shape

    # Test successful initialization
    geo_interdep = geometric_interdependence(
        b=b, m=m, interdependence_type='attribute',
        grid=grid, patch=patch,
        interdependence_matrix_mode='padding',
        device=device
    )
    xi_x = geo_interdep(x, device=device).view(x.size(0), -1, 81)
    torch.set_printoptions(threshold=torch.inf, linewidth=10000)
    print(geo_interdep.calculate_A().shape)
    print(xi_x.shape)
    print(xi_x[0, 200, :])
    print(xi_x[0, 201, :])
    print(xi_x[0, 202, :])



