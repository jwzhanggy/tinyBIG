# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################
# Test Basic Interdependence #
##############################

import pytest
import torch

from tinybig.interdependence import (
    constant_interdependence,
    constant_c_interdependence,
    zero_interdependence,
    one_interdependence,
    identity_interdependence
)


@pytest.fixture
def sample_data():
    b, m = 5, 3
    b_prime, m_prime = 4, 4
    A_instance = torch.randn(b, b_prime)
    A_attribute = torch.randn(m, m_prime)
    return b, m, b_prime, m_prime, A_instance, A_attribute


class Test_Constant_Interdependence:

    # Test cases for constant_interdependence
    def test_constant_interdependence_instance(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = constant_interdependence(b=b, m=m, A=A_instance, interdependence_type='instance')

        # Check if A is set correctly
        assert model.get_A().shape == A_instance.shape
        assert torch.allclose(model.get_A(), A_instance, atol=1e-6)

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        assert output.shape == (b_prime, m)

    def test_constant_interdependence_attribute(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = constant_interdependence(b=b, m=m, A=A_attribute, interdependence_type='attribute')

        # Check if A is set correctly
        assert model.get_A().shape == A_attribute.shape
        assert torch.allclose(model.get_A(), A_attribute, atol=1e-6)

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        assert output.shape == (b, m_prime)


class Test_Constant_C_Interdependence:

    # Test cases for constant_c_interdependence
    def test_constant_c_interdependence_instance(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = constant_c_interdependence(b=b, m=m, b_prime=b_prime, c=2.0, interdependence_type='instance')

        # Check if A is initialized with constant value
        assert torch.allclose(model.get_A(), 2.0 * torch.ones(b, b_prime))

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        assert output.shape == (b_prime, m)


    def test_constant_c_interdependence_attribute(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = constant_c_interdependence(b=b, m=m, m_prime=m_prime, c=2.0, interdependence_type='attribute')

        # Check if A is initialized with constant value
        assert torch.allclose(model.get_A(), 2.0 * torch.ones(m, m_prime))

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        assert output.shape == (b, m_prime)


class Test_Zero_Interdependence:

    # Test cases for zero_interdependence
    def test_zero_interdependence_instance(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = zero_interdependence(b=b, m=m, b_prime=b_prime, interdependence_type='instance')

        # Check if A is initialized with zero value
        assert torch.allclose(model.get_A(), torch.zeros(b, b_prime))

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        assert torch.allclose(output, torch.zeros(b_prime, m))


    def test_zero_interdependence_attribute(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = zero_interdependence(b=b, m=m, m_prime=m_prime, interdependence_type='attribute')

        # Check if A is initialized with zero value
        assert torch.allclose(model.get_A(), torch.zeros(m, m_prime))

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        assert torch.allclose(output, torch.zeros(b, m_prime))


class Test_One_Interdependence:

    # Test cases for one_interdependence
    def test_one_interdependence_instance(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = one_interdependence(b=b, m=m, b_prime=b_prime, interdependence_type='instance')

        # Check if A is initialized with one value
        assert torch.allclose(model.get_A(), torch.ones(b, b_prime))

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        assert torch.allclose(output, torch.matmul(torch.ones(b, b_prime).t(), x))


    def test_one_interdependence_attribute(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = one_interdependence(b=b, m=m, m_prime=m_prime, interdependence_type='attribute')

        # Check if A is initialized with one value
        assert torch.allclose(model.get_A(), torch.ones(m, m_prime))

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        assert torch.allclose(output, torch.matmul(x, torch.ones(m, m_prime)))


class Test_Identity_Interdependence:

    # Test cases for identity_interdependence
    def test_identity_interdependence_instance(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = identity_interdependence(b=b, m=m, b_prime=b_prime, interdependence_type='instance')

        # Check if A is initialized as identity matrix
        assert torch.allclose(model.get_A(), torch.eye(b, b_prime))

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        row_range = min(b, b_prime)
        assert torch.allclose(output[0:row_range,:], x[0:row_range,:])


    def test_identity_interdependence_attribute(self, sample_data):
        b, m, b_prime, m_prime, A_instance, A_attribute = sample_data
        model = identity_interdependence(b=b, m=m, m_prime=m_prime, interdependence_type='attribute')

        # Check if A is initialized as identity matrix
        assert torch.allclose(model.get_A(), torch.eye(m, m_prime))

        # Forward pass with a sample input
        x = torch.randn(b, m)
        output = model(x)
        column_range = min(m, m_prime)
        assert torch.allclose(output[:,0:column_range], x[:,0:column_range])


if __name__ == "__main__":
    pytest.main([__file__])
