# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

######################################
# Test Parameterized Interdependence #
######################################

import pytest
import torch

from tinybig.interdependence.parameterized_interdependence import (
    parameterized_interdependence,
    lowrank_parameterized_interdependence,
    hm_parameterized_interdependence,
    lphm_parameterized_interdependence,
    dual_lphm_parameterized_interdependence,
    random_matrix_adaption_parameterized_interdependence
)

# Factory fixture for creating an interdependence instance with dynamic parameters
@pytest.fixture
def interdependence_factory():
    def _create_interdependence(interdependence_class, b, m, **kwargs):
        # Instantiate the interdependence class with b and m and any extra kwargs
        interdep = interdependence_class(b=b, m=m, **kwargs)
        # Dynamically calculate l using the calculate_l method
        l = interdep.calculate_l()
        return interdep, l  # Return the instance and the dynamically calculated l
    return _create_interdependence

# Test case that uses the factory to dynamically calculate l and generate w
@pytest.mark.parametrize("interdependence_class, extra_params", [
    (parameterized_interdependence, {}),
    (lowrank_parameterized_interdependence, {"r": 2}),
    (hm_parameterized_interdependence, {"p": 1, "q": 1}),
    (lphm_parameterized_interdependence, {"r": 2, "p": 1, "q": 1}),
    (dual_lphm_parameterized_interdependence, {"r": 2, "p": 1, "q": 1}),
    (random_matrix_adaption_parameterized_interdependence, {"r": 2}),

    (parameterized_interdependence, {"interdependence_type": "instance"}),
    (lowrank_parameterized_interdependence, {"interdependence_type": "instance", "r": 2}),
    (hm_parameterized_interdependence, {"interdependence_type": "instance", "p": 1, "q": 1}),
    (lphm_parameterized_interdependence, {"interdependence_type": "instance", "r": 2, "p": 1, "q": 1}),
    (dual_lphm_parameterized_interdependence, {"interdependence_type": "instance", "r": 2, "p": 1, "q": 1}),
    (random_matrix_adaption_parameterized_interdependence, {"interdependence_type": "instance", "r": 2}),
])
@pytest.mark.parametrize("b, m", [(5, 3), (10, 7), (4, 4)])  # Test with different dimensions b, m
def test_interdependence_with_dynamic_l(interdependence_factory, interdependence_class, extra_params, b, m):
    """
    Test the interdependence classes with dynamic calculation of l based on b, m, and extra parameters.
    """
    # Create the interdependence instance and calculate l dynamically
    interdep, l = interdependence_factory(interdependence_class, b=b, m=m, **extra_params)

    # Generate a random weight matrix based on the calculated l
    w = torch.randn(1, l)
    assert w.shape == (1, l), f"Expected shape ({l},) but got {w.shape}"

    # Test the calculate_A method with the generated weight matrix
    A = interdep.calculate_A(w=w)
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (b, interdep.calculate_b_prime()) or A.shape == (m, interdep.calculate_m_prime()), "Matrix A shape mismatch"


# Low-rank specific test
@pytest.mark.parametrize("r", [1, 2, 3])
def test_lowrank_parameterized_interdependence(interdependence_factory, r):
    """
    Test the low-rank parameterized interdependence with different ranks r.
    """
    b, m = 5, 3
    interdep, l = interdependence_factory(lowrank_parameterized_interdependence, b=b, m=m, r=r)

    w = torch.randn(1, l)
    A = interdep.calculate_A(w=w)
    assert A.shape == (m, m), "Low-rank parameterized A matrix shape mismatch"


# HM specific test with p and q values
@pytest.mark.parametrize("p, q", [(1, 1), (3, 3), (1, 3)])
def test_hm_parameterized_interdependence(interdependence_factory, p, q):
    """
    Test the HM parameterized interdependence with different p and q values.
    """
    b, m = 5, 3
    interdep, l = interdependence_factory(hm_parameterized_interdependence, b=b, m=m, p=p, q=q)

    w = torch.randn(1, l)
    A = interdep.calculate_A(w=w)
    assert A.shape == (m, m), "HM parameterized A matrix shape mismatch"


# Other interdependence classes are similar and follow the same pattern as above
