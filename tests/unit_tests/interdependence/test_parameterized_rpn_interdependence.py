# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################################
# Test Parameterized RPN Interdependence #
##########################################


import pytest
import torch

from tinybig.interdependence.parameterized_rpn_interdependence import parameterized_rpn_interdependence
from tinybig.expansion import taylor_expansion, identity_expansion
from tinybig.reconciliation import identity_reconciliation, lorr_reconciliation

# Factory fixture for creating interdependence instances with dynamic data transformation and parameter fabrication
@pytest.fixture
def interdependence_factory():
    def _create_interdependence(interdependence_class, b, m, x, data_transformation, parameter_fabrication, **kwargs):
        # Instantiate the interdependence class with b, m, data transformation, and parameter fabrication
        interdep = interdependence_class(b=b, m=m, x=x, data_transformation=data_transformation,
                                         parameter_fabrication=parameter_fabrication, **kwargs)
        # Dynamically calculate l using the calculate_l method
        l = interdep.calculate_l()
        return interdep, l  # Return the instance and the calculated l

    return _create_interdependence


# Test case that uses the factory to dynamically calculate l, generate w and x (data batch)
@pytest.mark.parametrize("data_transformation, parameter_fabrication", [
    (taylor_expansion(), lorr_reconciliation()),
    (identity_expansion(), identity_reconciliation()),
    # Add more transformation and fabrication combinations as needed
])
@pytest.mark.parametrize("b, m", [(5, 3), (10, 7), (4, 4)])  # Test with different dimensions b, m
def test_rpn_interdependence_with_data_batch(interdependence_factory, data_transformation, parameter_fabrication, b, m):
    """
    Test the RPN interdependence classes with dynamic data transformation, parameter fabrication, and dynamic calculation of l based on b and m.
    """
    # Create a random data batch x
    x = torch.randn(b, m)

    # Create the interdependence instance and calculate l dynamically
    interdep, l = interdependence_factory(parameterized_rpn_interdependence, b=b, m=m, x=x,
                                          data_transformation=data_transformation,
                                          parameter_fabrication=parameter_fabrication)

    # Generate a random weight matrix based on the calculated l
    w = torch.randn(1, l)  # Assuming l x l for RPN interdependence
    assert w.shape == (1, l), f"Expected weight matrix shape ({l}, {l}) but got {w.shape}"

    # Test the calculate_A method with the generated weight matrix and data batch
    A = interdep.calculate_A(x=x, w=w)
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (b, b) or A.shape == (m, m), "Matrix A shape mismatch"


# Specific tests for Transformation and Fabrication combinations
@pytest.mark.parametrize("data_transformation, parameter_fabrication", [
    (taylor_expansion(), lorr_reconciliation()),
    (identity_expansion(), identity_reconciliation()),
])
def test_specific_rpn_interdependence(interdependence_factory, data_transformation, parameter_fabrication):
    """
    Test specific combinations of data transformation and parameter fabrication functions.
    """
    b, m = 5, 3
    x = torch.randn(b, m)
    interdep, l = interdependence_factory(parameterized_rpn_interdependence, b=b, m=m, x=x,
                                          data_transformation=data_transformation,
                                          parameter_fabrication=parameter_fabrication)

    w = torch.randn(1, l)
    A = interdep.calculate_A(x=x, w=w)
    assert A.shape == (b, b) or A.shape == (m, m), "Matrix A shape mismatch"


# Exception handling tests
@pytest.mark.parametrize("data_transformation, parameter_fabrication", [
    (None, identity_reconciliation),  # Missing data transformation
    (identity_expansion, None),  # Missing parameter fabrication
])
def test_rpn_interdependence_exceptions(interdependence_factory, data_transformation, parameter_fabrication):
    """
    Test exception handling for RPN interdependence classes when data transformation or parameter fabrication is missing.
    """
    b, m = 5, 3
    x = torch.randn(b, m)

    # Test for missing transformation or fabrication (should raise a ValueError)
    with pytest.raises(ValueError):
        interdependence_factory(parameterized_rpn_interdependence, b=b, m=m, x=x,
                                data_transformation=data_transformation, parameter_fabrication=parameter_fabrication)




