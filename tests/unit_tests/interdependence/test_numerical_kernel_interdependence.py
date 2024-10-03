# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#########################################
# Test Numerical Kernel Interdependence #
#########################################

import pytest
import torch
import numpy as np

from tinybig.interdependence.numerical_kernel_interdependence import (
    numerical_kernel_based_interdependence,
    linear_kernel_interdependence,
    cosine_similarity_interdependence,
    minkowski_distance_interdependence,
    manhattan_distance_interdependence,
    euclidean_distance_interdependence,
    chebyshev_distance_interdependence,
    canberra_distance_interdependence,
    polynomial_kernel_interdependence,
    hyperbolic_tangent_kernel_interdependence,
    exponential_kernel_interdependence,
    gaussian_rbf_kernel_interdependence,
    laplacian_kernel_interdependence,
    anisotropic_rbf_kernel_interdependence,
    custom_hybrid_kernel_interdependence,
)


@pytest.fixture
def sample_data():
    b, m = 5, 3
    X = torch.randn(b, m)
    return b, m, X


@pytest.mark.parametrize("kernel_class", [
    linear_kernel_interdependence,
    cosine_similarity_interdependence,
    manhattan_distance_interdependence,
    euclidean_distance_interdependence,
    chebyshev_distance_interdependence,
    canberra_distance_interdependence,
])
def test_basic_metric_interdependence(sample_data, kernel_class):
    """
    Test basic metric-based interdependence classes like linear, cosine, manhattan, etc.
    """
    b, m, X = sample_data
    interdep = kernel_class(b=b, m=m, device='cpu')

    # Ensure the A matrix is computed and has the correct shape
    A = interdep.calculate_A(x=X)
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (m, m), "Matrix A shape mismatch"


@pytest.mark.parametrize("kernel_class, kernel_args", [
    (minkowski_distance_interdependence, {"p": 2}),
    (polynomial_kernel_interdependence, {"c": 1, "d": 2}),
    (hyperbolic_tangent_kernel_interdependence, {"c": 0.1, "alpha": 0.5}),
    (exponential_kernel_interdependence, {"gamma": 1.0}),
    (gaussian_rbf_kernel_interdependence, {"sigma": 1.0}),
    (laplacian_kernel_interdependence, {"sigma": 1.0}),
])
def test_advanced_metric_interdependence(sample_data, kernel_class, kernel_args):
    """
    Test more advanced metric-based interdependence classes like Minkowski, polynomial, exponential, etc.
    """
    b, m, X = sample_data
    interdep = kernel_class(b=b, m=m, interdependence_type='attribute', device='cpu', **kernel_args)

    # Ensure the A matrix is computed and has the correct shape
    A = interdep.calculate_A(x=X)
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (m, m), "Matrix A shape mismatch"


def test_anisotropic_rbf_kernel_interdependence(sample_data):
    """
    Test anisotropic RBF kernel with both vector and scalar input.
    """
    b, m, X = sample_data
    a_vector = torch.ones(b)
    a_scalar = 1.0

    # Test vector case
    interdep_vector = anisotropic_rbf_kernel_interdependence(b=b, m=m, interdependence_type='attribute', a_vector=a_vector, device='cpu')
    A_vector = interdep_vector.calculate_A(x=X)
    assert A_vector.shape == (m, m), "Anisotropic RBF kernel (vector) shape mismatch"

    # Test scalar case
    interdep_scalar = anisotropic_rbf_kernel_interdependence(b=b, m=m, interdependence_type='attribute', a_scalar=a_scalar, device='cpu')
    A_scalar = interdep_scalar.calculate_A(x=X)
    assert A_scalar.shape == (m, m), "Anisotropic RBF kernel (scalar) shape mismatch"


def test_custom_hybrid_kernel_interdependence(sample_data):
    """
    Test custom hybrid kernel with a combination of multiple kernels.
    """
    b, m, X = sample_data

    # Define two simple kernels for testing purposes
    def kernel_1(x, *args, **kwargs):
        x = x.T
        return torch.exp(-torch.cdist(x, x))

    def kernel_2(x, *args, **kwargs):
        x = x.T
        return 1 / (1 + torch.cdist(x, x))

    kernels = [kernel_1, kernel_2]
    weights = [0.5, 0.5]

    interdep = custom_hybrid_kernel_interdependence(b=b, m=m, kernels=kernels, weights=weights, device='cpu')

    # Ensure the A matrix is computed and has the correct shape
    A = interdep.calculate_A(x=X)
    assert A.shape == (m, m), "Custom hybrid kernel matrix shape mismatch"


@pytest.mark.parametrize("kernel_class", [
    linear_kernel_interdependence,
    cosine_similarity_interdependence,
    manhattan_distance_interdependence,
])
def test_kernel_interdependence_exceptions(sample_data, kernel_class):
    """
    Test exception handling for metric-based interdependence classes when required data is not provided.
    """
    b, m, _ = sample_data

    # Test without data (should raise an AssertionError)
    interdep = kernel_class(b=b, m=m, device='cpu')
    with pytest.raises(AssertionError):
        interdep.calculate_A()

