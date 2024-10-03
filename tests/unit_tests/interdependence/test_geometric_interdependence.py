# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Test Geometric Interdependence #
##################################

import pytest
from tinybig.interdependence.geometric_interdependence import geometric_interdependence
from tinybig.koala.geometry import grid as grid_structure
from tinybig.koala.geometry import cuboid, sphere, cylinder


@pytest.fixture
def sample_grid_and_patch():
    # Create a grid and different patch types
    grid = grid_structure(h=10, w=10, d=5)
    patch1 = cuboid(p_h=2, p_w=2, p_d=2)
    patch2 = sphere(p_r=2)
    patch3 = cylinder(p_r=2, p_d=3)
    return grid, patch1, patch2, patch3


@pytest.mark.parametrize("patch_type", ['cuboid', 'sphere', 'cylinder'])
def test_geometric_interdependence_initialization(sample_grid_and_patch, patch_type):
    """
    Test the initialization of geometric_interdependence class with different patches.
    """
    grid, patch1, patch2, patch3 = sample_grid_and_patch

    # Select patch based on patch_type
    if patch_type == 'cuboid':
        patch = patch1
    elif patch_type == 'sphere':
        patch = patch2
    elif patch_type == 'cylinder':
        patch = patch3

    b, m = 10, 500

    # Test successful initialization
    geo_interdep = geometric_interdependence(
        b=b, m=m, interdependence_type='attribute',
        grid=grid, patch=patch,
        interdependence_matrix_mode='padding', normalization=True
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


@pytest.mark.parametrize("patch_type", ['cuboid', 'sphere', 'cylinder'])
def test_geometric_interdependence_calculations(sample_grid_and_patch, patch_type):
    """
    Test calculations of b_prime, m_prime, and grid shape related methods.
    """
    grid, patch1, patch2, patch3 = sample_grid_and_patch

    if patch_type == 'cuboid':
        patch = patch1
    elif patch_type == 'sphere':
        patch = patch2
    elif patch_type == 'cylinder':
        patch = patch3

    b, m = 10, 500
    geo_interdep = geometric_interdependence(
        b=b, m=m, interdependence_type='attribute',
        grid=grid, patch=patch,
        interdependence_matrix_mode='padding', normalization=True,
        cd_h=2, cd_w=2, cd_d=2,
    )

    b_prime = geo_interdep.calculate_b_prime()
    m_prime = geo_interdep.calculate_m_prime()

    assert b_prime > 0, "b_prime should be greater than 0"
    assert m_prime > 0, "m_prime should be greater than 0"
    assert geo_interdep.get_patch_size() == patch.get_volume()
    assert geo_interdep.get_patch_num() == grid.get_patch_num(
        cd_h=geo_interdep.cd_h, cd_w=geo_interdep.cd_w, cd_d=geo_interdep.cd_d
    )
    assert geo_interdep.get_grid_shape() == grid.get_grid_shape()
    assert geo_interdep.get_grid_shape_after_packing() == grid.get_grid_shape_after_packing(
        cd_h=geo_interdep.cd_h, cd_w=geo_interdep.cd_w, cd_d=geo_interdep.cd_d
    )


@pytest.mark.parametrize("patch_type", ['cuboid', 'sphere', 'cylinder'])
def test_geometric_interdependence_matrix(sample_grid_and_patch, patch_type):
    """
    Test the matrix A calculation for geometric interdependence.
    """
    grid, patch1, patch2, patch3 = sample_grid_and_patch

    if patch_type == 'cuboid':
        patch = patch1
    elif patch_type == 'sphere':
        patch = patch2
    elif patch_type == 'cylinder':
        patch = patch3

    b, m = 500, 10
    geo_interdep = geometric_interdependence(
        b=b, m=m, interdependence_type='instance',
        grid=grid, patch=patch,
        interdependence_matrix_mode='aggregation', normalization=False
    )

    A = geo_interdep.calculate_A()
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (b, geo_interdep.calculate_b_prime()), "A shape does not match expected shape"

    # Check normalization and matrix mode (padding)
    geo_interdep_norm = geometric_interdependence(
        b=b, m=m, interdependence_type='instance',
        grid=grid, patch=patch,
        interdependence_matrix_mode='padding', normalization=True
    )
    A_norm = geo_interdep_norm.calculate_A()
    assert A_norm.shape == (b, geo_interdep_norm.calculate_b_prime()), "Normalized matrix A shape mismatch"


@pytest.mark.parametrize("patch_type", ['cuboid', 'sphere', 'cylinder'])
def test_geometric_interdependence_update_methods(sample_grid_and_patch, patch_type):
    """
    Test the update methods for grid, patch, and packing strategy.
    """
    grid, patch1, patch2, patch3 = sample_grid_and_patch

    if patch_type == 'cuboid':
        patch = patch1
    elif patch_type == 'sphere':
        patch = patch2
    elif patch_type == 'cylinder':
        patch = patch3

    b, m = 500, 10
    geo_interdep = geometric_interdependence(
        b=b, m=m, interdependence_type='instance',
        grid=grid, patch=patch
    )

    # Test updating grid
    new_grid = grid_structure(h=20, w=20, d=20)
    geo_interdep.update_grid(new_grid)
    assert geo_interdep.grid == new_grid, "Grid update failed"

    # Test updating patch
    new_patch = cuboid(p_h=3, p_w=3, p_d=3)
    geo_interdep.update_patch(new_patch)
    assert geo_interdep.patch == new_patch, "Patch update failed"

    # Test updating packing strategy
    new_packing_strategy = 'different_packing'
    geo_interdep.update_packing_strategy(new_packing_strategy)
    assert geo_interdep.packing_strategy == new_packing_strategy, "Packing strategy update failed"


@pytest.mark.parametrize("patch_type", ['cuboid', 'sphere', 'cylinder'])
def test_geometric_interdependence_edge_cases(sample_grid_and_patch, patch_type):
    """
    Test edge cases for geometric interdependence, including small patches or large grids.
    """
    grid, patch1, patch2, patch3 = sample_grid_and_patch

    if patch_type == 'cuboid':
        patch = patch1
    elif patch_type == 'sphere':
        patch = patch2
    elif patch_type == 'cylinder':
        patch = patch3
    else:
        raise ValueError("Invalid patch type")

    b, m = 1, 500
    geo_interdep = geometric_interdependence(
        b=b, m=m, interdependence_type='attribute',
        grid=grid, patch=patch,
        interdependence_matrix_mode='aggregation', normalization=True
    )

    assert geo_interdep.calculate_b_prime() == 1, "For b=1, b_prime should be 1"
    assert geo_interdep.calculate_m_prime() == 500, "For m=500, m_prime should be 500"

    # Test grid with very small patch
    small_patch = cuboid(p_h=1, p_w=1, p_d=1)
    geo_interdep_small_patch = geometric_interdependence(
        b=b, m=m, interdependence_type='attribute',
        grid=grid, patch=small_patch
    )
    assert geo_interdep_small_patch.calculate_b_prime() > 0, "b_prime should be greater than 0 for small patch"


@pytest.mark.parametrize("patch_type", ['cuboid', 'sphere', 'cylinder'])
def test_geometric_interdependence_exceptions(sample_grid_and_patch, patch_type):
    """
    Test exception cases, including invalid grid and patch configurations.
    """
    grid, patch1, patch2, patch3 = sample_grid_and_patch

    if patch_type == 'cuboid':
        patch = patch1
    elif patch_type == 'sphere':
        patch = patch2
    elif patch_type == 'cylinder':
        patch = patch3

    b, m = 10, 500

    # Invalid grid (None)
    with pytest.raises(ValueError):
        geometric_interdependence(
            b=b, m=m, interdependence_type='attribute',
            grid=None, patch=patch
        )

    # Invalid patch (None)
    with pytest.raises(ValueError):
        geometric_interdependence(
            b=b, m=m, interdependence_type='attribute',
            grid=grid, patch=None
        )
