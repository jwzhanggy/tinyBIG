# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################################
# Test Statistical Kernel Interdependence #
###########################################


import pytest
import torch

from tinybig.interdependence.statistical_kernel_interdependence import (
    statistical_kernel_based_interdependence,
    kl_divergence_interdependence,
    pearson_correlation_interdependence,
    rv_coefficient_interdependence,
    mutual_information_interdependence
)


@pytest.fixture
def sample_data():
    b, m = 5, 3
    X = torch.randn(b, m)
    return b, m, X


@pytest.mark.parametrize("kernel_class", [
    kl_divergence_interdependence,
    pearson_correlation_interdependence,
    rv_coefficient_interdependence,
    mutual_information_interdependence,
])
def test_statistical_kernel_interdependence(sample_data, kernel_class):
    """
    Test statistical kernel-based interdependence classes such as KL divergence, Pearson correlation, RV coefficient, and Mutual Information.
    """
    b, m, X = sample_data
    interdep = kernel_class(b=b, m=m, device='cpu')

    # Ensure the A matrix is computed and has the correct shape
    A = interdep.calculate_A(x=X)
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (m, m), "Matrix A shape mismatch"


def test_kl_divergence_interdependence(sample_data):
    """
    Test KL Divergence interdependence kernel.
    """
    b, m, X = sample_data
    interdep = kl_divergence_interdependence(b=b, m=m, device='cpu')

    # Ensure the A matrix is computed and has the correct shape
    A = interdep.calculate_A(x=X)
    assert A.shape == (m, m), "KL divergence A matrix shape mismatch"


def test_pearson_correlation_interdependence(sample_data):
    """
    Test Pearson correlation interdependence kernel.
    """
    b, m, X = sample_data
    interdep = pearson_correlation_interdependence(b=b, m=m, device='cpu')

    # Ensure the A matrix is computed and has the correct shape
    A = interdep.calculate_A(x=X)
    assert A.shape == (m, m), "Pearson correlation A matrix shape mismatch"


def test_rv_coefficient_interdependence(sample_data):
    """
    Test RV Coefficient interdependence kernel.
    """
    b, m, X = sample_data
    interdep = rv_coefficient_interdependence(b=b, m=m, device='cpu')

    # Ensure the A matrix is computed and has the correct shape
    A = interdep.calculate_A(x=X)
    assert A.shape == (m, m), "RV coefficient A matrix shape mismatch"


def test_mutual_information_interdependence(sample_data):
    """
    Test Mutual Information interdependence kernel.
    """
    b, m, X = sample_data
    interdep = mutual_information_interdependence(b=b, m=m, device='cpu')

    # Ensure the A matrix is computed and has the correct shape
    A = interdep.calculate_A(x=X)
    assert A.shape == (m, m), "Mutual Information A matrix shape mismatch"


@pytest.mark.parametrize("kernel_class", [
    kl_divergence_interdependence,
    pearson_correlation_interdependence,
    rv_coefficient_interdependence,
    mutual_information_interdependence,
])
def test_statistical_kernel_interdependence_exceptions(sample_data, kernel_class):
    """
    Test exception handling for statistical kernel-based interdependence classes when required data is not provided.
    """
    b, m, _ = sample_data

    # Test without data (should raise an AssertionError)
    interdep = kernel_class(b=b, m=m, device='cpu')
    with pytest.raises(AssertionError):
        interdep.calculate_A()

