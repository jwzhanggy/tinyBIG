# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################################
# Test Parameterized Bilinear Interdependence #
###############################################


import pytest
import torch

from tinybig.interdependence.parameterized_bilinear_interdependence import (
    parameterized_bilinear_interdependence,
    lowrank_parameterized_bilinear_interdependence,
    hm_parameterized_bilinear_interdependence,
    lphm_parameterized_bilinear_interdependence,
    dual_lphm_parameterized_bilinear_interdependence,
    random_matrix_adaption_parameterized_bilinear_interdependence
)


# Fixture for creating interdependence instances with both batch data and parameters
@pytest.fixture
def interdependence_factory():
    def _create_interdependence(interdependence_class, b, m, x, **kwargs):
        # Instantiate the interdependence class with b, m, and pass other parameters
        interdep = interdependence_class(b=b, m=m, **kwargs)
        # Dynamically calculate l using the calculate_l method
        l = interdep.calculate_l()
        return interdep, l  # Return the instance and the calculated l

    return _create_interdependence


# Test case that uses the factory to dynamically calculate l, generate w and x (data batch)
@pytest.mark.parametrize("interdependence_class, extra_params", [
    (parameterized_bilinear_interdependence, {}),
    (lowrank_parameterized_bilinear_interdependence, {"r": 2}),
    (hm_parameterized_bilinear_interdependence, {"p": 1, "q": 1}),
    (lphm_parameterized_bilinear_interdependence, {"r": 2, "p": 1, "q": 1}),
    (dual_lphm_parameterized_bilinear_interdependence, {"r": 2, "p": 1, "q": 1}),
    (random_matrix_adaption_parameterized_bilinear_interdependence, {"r": 2}),

    (parameterized_bilinear_interdependence, {"interdependence_type": "instance"}),
    (lowrank_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "r": 2}),
    (hm_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "p": 1, "q": 1}),
    (lphm_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "r": 2, "p": 1, "q": 1}),
    (dual_lphm_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "r": 2, "p": 1, "q": 1}),
    (random_matrix_adaption_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "r": 2}),
])
@pytest.mark.parametrize("b, m", [(5, 3), (10, 7), (4, 4)])  # Test with different dimensions b, m
def test_bilinear_interdependence_with_data_batch(interdependence_factory, interdependence_class, extra_params, b, m):
    """
    Test the bilinear interdependence classes with data batch and dynamic calculation of l based on b and m.
    """
    # Create a random data batch x
    x = torch.randn(b, m)

    # Create the interdependence instance and calculate l dynamically
    interdep, l = interdependence_factory(interdependence_class, b=b, m=m, x=x, **extra_params)

    # Generate a random weight matrix based on the calculated l
    w = torch.randn(1, l)  # Assuming l x l for bilinear interdependence
    assert w.shape == (1, l), f"Expected weight matrix shape ({l}, {l}) but got {w.shape}"

    # Test the calculate_A method with the generated weight matrix and data batch
    if interdep.interdependence_type == "instance":
        A = interdep.calculate_A(x=x.t(), w=w)
    else:
        A = interdep.calculate_A(x=x, w=w)
    assert A is not None, "Matrix A should not be None"
    assert (A.shape == (b, b) and interdep.interdependence_type == "instance") or (A.shape == (m, m) and interdep.interdependence_type == "attribute"), "Matrix A shape mismatch"


# Specific tests for low-rank bilinear parameterized interdependence
@pytest.mark.parametrize("r", [1, 2, 3])
def test_lowrank_bilinear_interdependence(interdependence_factory, r):
    """
    Test the low-rank parameterized bilinear interdependence with different ranks r.
    """
    b, m = 5, 3
    x = torch.randn(b, m)
    interdep, l = interdependence_factory(lowrank_parameterized_bilinear_interdependence, b=b, m=m, x=x, r=r)

    w = torch.randn(1, l)
    if interdep.interdependence_type == "instance":
        A = interdep.calculate_A(x=x.t(), w=w)
    else:
        A = interdep.calculate_A(x=x, w=w)
    assert (A.shape == (b, b) and interdep.interdependence_type == "instance") or (A.shape == (m, m) and interdep.interdependence_type == "attribute"), "Matrix A shape mismatch"


# Specific tests for HM parameterized bilinear interdependence
@pytest.mark.parametrize("p, q", [(1, 1), (1, 5), (5, 1), (5, 5)])
def test_hm_bilinear_interdependence(interdependence_factory, p, q):
    """
    Test the HM parameterized bilinear interdependence with different p and q values.
    """
    b, m = 3, 5
    x = torch.randn(b, m)
    interdep, l = interdependence_factory(hm_parameterized_bilinear_interdependence, b=b, m=m, x=x, p=p, q=q)

    w = torch.randn(1, l)
    if interdep.interdependence_type == "instance":
        A = interdep.calculate_A(x=x.t(), w=w)
    else:
        A = interdep.calculate_A(x=x, w=w)
    assert (A.shape == (b, b) and interdep.interdependence_type == "instance") or (A.shape == (m, m) and interdep.interdependence_type == "attribute"), "Matrix A shape mismatch"


# Specific tests for LPHM parameterized bilinear interdependence
@pytest.mark.parametrize("r, p, q", [(1, 1, 1), (2, 5, 5)])
def test_lphm_bilinear_interdependence(interdependence_factory, r, p, q):
    """
    Test the LPHM parameterized bilinear interdependence with different ranks and block sizes.
    """
    b, m = 3, 5
    x = torch.randn(b, m)
    interdep, l = interdependence_factory(lphm_parameterized_bilinear_interdependence, b=b, m=m, x=x, r=r, p=p, q=q)

    w = torch.randn(1, l)
    if interdep.interdependence_type == "instance":
        A = interdep.calculate_A(x=x.t(), w=w)
    else:
        A = interdep.calculate_A(x=x, w=w)
    assert (A.shape == (b, b) and interdep.interdependence_type == "instance") or (A.shape == (m, m) and interdep.interdependence_type == "attribute"), "Matrix A shape mismatch"


# Specific tests for Dual LPHM parameterized bilinear interdependence
@pytest.mark.parametrize("r, p, q", [(1, 1, 1), (2, 5, 5)])
def test_dual_lphm_bilinear_interdependence(interdependence_factory, r, p, q):
    """
    Test the dual LPHM parameterized bilinear interdependence with different ranks and block sizes.
    """
    b, m = 3, 5
    x = torch.randn(b, m)
    interdep, l = interdependence_factory(dual_lphm_parameterized_bilinear_interdependence, b=b, m=m, x=x, r=r, p=p,
                                          q=q)

    w = torch.randn(1, l)
    if interdep.interdependence_type == "instance":
        A = interdep.calculate_A(x=x.t(), w=w)
    else:
        A = interdep.calculate_A(x=x, w=w)
    assert (A.shape == (b, b) and interdep.interdependence_type == "instance") or (A.shape == (m, m) and interdep.interdependence_type == "attribute"), "Matrix A shape mismatch"


# Specific tests for Random Matrix Adaption parameterized bilinear interdependence
@pytest.mark.parametrize("r", [1, 2, 3])
def test_random_matrix_adaption_bilinear_interdependence(interdependence_factory, r):
    """
    Test the random matrix adaption parameterized bilinear interdependence with different ranks.
    """
    b, m = 5, 3
    x = torch.randn(b, m)
    interdep, l = interdependence_factory(random_matrix_adaption_parameterized_bilinear_interdependence, b=b, m=m, x=x,
                                          r=r)

    w = torch.randn(1, l)
    if interdep.interdependence_type == "instance":
        A = interdep.calculate_A(x=x.t(), w=w)
    else:
        A = interdep.calculate_A(x=x, w=w)
    assert (A.shape == (b, b) and interdep.interdependence_type == "instance") or (A.shape == (m, m) and interdep.interdependence_type == "attribute"), "Matrix A shape mismatch"


# Exception handling tests
@pytest.mark.parametrize("interdependence_class, extra_params", [
    (parameterized_bilinear_interdependence, {}),
    (lowrank_parameterized_bilinear_interdependence, {"r": 2}),
    (hm_parameterized_bilinear_interdependence, {"p": 1, "q": 1}),
    (lphm_parameterized_bilinear_interdependence, {"r": 2, "p": 1, "q": 1}),
    (dual_lphm_parameterized_bilinear_interdependence, {"r": 2, "p": 1, "q": 1}),
    (random_matrix_adaption_parameterized_bilinear_interdependence, {"r": 2}),
    ])
def test_bilinear_interdependence_exceptions(interdependence_factory, interdependence_class, extra_params):
    """
    Test exception handling for bilinear interdependence classes when required data or parameters are missing.
    """
    b, m = 5, 3
    interdep, _ = interdependence_factory(interdependence_class, b=b, m=m, x=None, **extra_params)

    # Test with missing weight matrix (should raise an AssertionError)
    with pytest.raises(AssertionError):
        interdep.calculate_A()



