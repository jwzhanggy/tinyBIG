# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

########################################
# Test Kernels in koala.linear_algebra #
########################################

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../tinybig')))

import pytest
import torch

from tinybig.koala.linear_algebra import (
    linear_kernel, polynomial_kernel, hyperbolic_tangent_kernel, cosine_similarity_kernel,
    exponential_kernel, gaussian_rbf_kernel, laplacian_kernel, anisotropic_rbf_kernel,
    minkowski_distance, manhattan_distance, euclidean_distance,
    chebyshev_distance, canberra_distance,
    minkowski_distance_kernel, manhattan_distance_kernel, euclidean_distance_kernel,
    chebyshev_distance_kernel, canberra_distance_kernel,
    custom_hybrid_kernel
)


class Test_Instance_Kernel:
    @pytest.mark.parametrize("kernel_func, x1, x2, expected, args, kwargs", [
        (linear_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (linear_kernel, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (linear_kernel, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (linear_kernel, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (linear_kernel, torch.tensor([]), torch.tensor([]), ValueError, (), {}),

        (polynomial_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (polynomial_kernel, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (polynomial_kernel, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (polynomial_kernel, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (polynomial_kernel, torch.tensor([]), torch.tensor([]), ValueError, (), {}),

        (hyperbolic_tangent_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (hyperbolic_tangent_kernel, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (hyperbolic_tangent_kernel, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (hyperbolic_tangent_kernel, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (hyperbolic_tangent_kernel, torch.tensor([]), torch.tensor([]), ValueError, (), {}),

        (cosine_similarity_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (cosine_similarity_kernel, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (cosine_similarity_kernel, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (cosine_similarity_kernel, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (cosine_similarity_kernel, torch.tensor([]), torch.tensor([]), ValueError, (), {}),

        (exponential_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (exponential_kernel, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (exponential_kernel, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (exponential_kernel, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (exponential_kernel, torch.tensor([]), torch.tensor([]), ValueError, (), {}),

        (gaussian_rbf_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (gaussian_rbf_kernel, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (gaussian_rbf_kernel, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (gaussian_rbf_kernel, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (gaussian_rbf_kernel, torch.tensor([]), torch.tensor([]), ValueError, (), {}),
        (gaussian_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 5.0]), ValueError, (), {'sigma': 0.0}),
        (gaussian_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 5.0]), ValueError, (), {'sigma': -1.0}),

        (laplacian_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (laplacian_kernel, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (laplacian_kernel, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (laplacian_kernel, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (laplacian_kernel, torch.tensor([]), torch.tensor([]), ValueError, (), {}),
        (laplacian_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 5.0]), ValueError, (), {'sigma': 0.0}),

        (anisotropic_rbf_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (anisotropic_rbf_kernel, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (anisotropic_rbf_kernel, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (anisotropic_rbf_kernel, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (anisotropic_rbf_kernel, torch.tensor([]), torch.tensor([]), ValueError, (), {}),
        (anisotropic_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), ValueError, (), {'a_vector': torch.tensor([0.0, 0.0, 0.0])}),

        (minkowski_distance, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {'p': 1}),
        (minkowski_distance, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {'p': 1}),
        (minkowski_distance, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {'p': 1}),
        (minkowski_distance, torch.tensor([]), torch.tensor(1.0), ValueError, (), {'p': 1}),
        (minkowski_distance, torch.tensor([]), torch.tensor([]), ValueError, (), {'p': 1}),
        (minkowski_distance, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), ValueError, (), {}),
        (minkowski_distance, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), ValueError, (), {'p': 'inf'}),

        (manhattan_distance, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (manhattan_distance, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (manhattan_distance, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (manhattan_distance, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (manhattan_distance, torch.tensor([]), torch.tensor([]), ValueError, (), {}),

        (euclidean_distance, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (euclidean_distance, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (euclidean_distance, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (euclidean_distance, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (euclidean_distance, torch.tensor([]), torch.tensor([]), ValueError, (), {}),

        (chebyshev_distance, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (chebyshev_distance, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (chebyshev_distance, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (chebyshev_distance, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (chebyshev_distance, torch.tensor([]), torch.tensor([]), ValueError, (), {}),

        (canberra_distance, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]), ValueError, (), {}),
        (canberra_distance, torch.tensor([[1.0, 2.0]]), torch.tensor([5.0, 6.0]), ValueError, (), {}),
        (canberra_distance, None, torch.tensor([[1.0, 2.0]]), ValueError, (), {}),
        (canberra_distance, torch.tensor([]), torch.tensor(1.0), ValueError, (), {}),
        (canberra_distance, torch.tensor([]), torch.tensor([]), ValueError, (), {}),

        ###############################################################################################################

        (linear_kernel, torch.tensor([1.0, 2.0]), torch.tensor([1.0, -2.0]), torch.tensor(-3.0), (), {}),
        (linear_kernel, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor(0.0), (), {}),
        (linear_kernel, torch.tensor([-1.0, -2.0]), torch.tensor([-1.0, -2.0]), torch.tensor(5.0), (), {}),

        (polynomial_kernel, torch.tensor([1.0, 2.0]), torch.tensor([1.0, -2.0]), ValueError, (), {'c': 3.0, 'd': -2}),
        (polynomial_kernel, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), ValueError, (), {'d': -1}),
        (polynomial_kernel, torch.tensor([-1.0, -2.0]), torch.tensor([-1.0, -2.0]), torch.tensor(8.0), (), {'c': 3.0}),

        (hyperbolic_tangent_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tanh(torch.tensor(12.0)), (), {'alpha': 1.0, 'c': 1.0}),
        (hyperbolic_tangent_kernel, torch.tensor([1.0, -2.0]), torch.tensor([-3.0, 2.0]), torch.tanh(torch.tensor(-3.0)), (), {'alpha': 0.5, 'c': 0.5}),
        (hyperbolic_tangent_kernel, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tanh(torch.tensor(1.0)), (), {'alpha': 1.0, 'c': 1.0}),
        (hyperbolic_tangent_kernel, torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]), torch.tanh(torch.tensor(3.0)), (), {'alpha': 2.0, 'c': -1.0}),
        (hyperbolic_tangent_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tanh(torch.tensor(16.0)), (), {'alpha': 0.5, 'c': 0.0}),

        (cosine_similarity_kernel, torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor(1.0), (), {}),
        (cosine_similarity_kernel, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor(0.0), (), {}),
        (cosine_similarity_kernel, torch.tensor([1.0, -1.0]), torch.tensor([1.0, -1.0]), torch.tensor(1.0), (), {}),
        (cosine_similarity_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([7.0, 8.0, 9.0]), torch.tensor(0.95941194556), (), {}),

        (minkowski_distance, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor(0.0), (), {'p': 1}),
        (minkowski_distance, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.tensor(7.0), (), {'p': 1}),
        (minkowski_distance, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.tensor(5.0), (), {'p': 2}),
        (minkowski_distance, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.tensor(4.4979), (), {'p': 3}),
        (minkowski_distance, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.tensor(4.28457229495), (), {'p': 4}),
        (minkowski_distance, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.tensor(4.0), (), {'p': torch.inf}),

        (manhattan_distance, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor(0.0), (), {}),
        (manhattan_distance, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.tensor(7.0), (), {}),
        (manhattan_distance, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor(0.0), (), {}),
        (manhattan_distance, torch.tensor([1.0, -2.0]), torch.tensor([-3.0, 4.0]), torch.tensor(10.0), (), {}),
        (manhattan_distance, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 6.0, 8.0]), torch.tensor(12.0), (), {}),

        (euclidean_distance, torch.tensor([0.0, 1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]), torch.tensor(5.1961524227), (), {}),
        (euclidean_distance, torch.tensor([1.0]), torch.tensor([4.0]), torch.tensor(3.0), (), {}),
        (euclidean_distance, torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, -1.0]), torch.tensor(2.828427124), (), {}),
        (euclidean_distance, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor(0.0), (), {}),
        (euclidean_distance, torch.tensor([1.5, -2.7, 3.14]), torch.tensor([-1.5, 2.7, -3.14]), torch.tensor(8.80899540243), (), {}),

        (chebyshev_distance, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor(0.0), (), {}),
        (chebyshev_distance, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.tensor(4.0), (), {}),
        (chebyshev_distance, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor(0.0), (), {}),
        (chebyshev_distance, torch.tensor([1.0, -2.0]), torch.tensor([-3.0, 4.0]), torch.tensor(6.0), (), {}),
        (chebyshev_distance, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 6.0, 9.0]), torch.tensor(6.0), (), {}),

        (canberra_distance, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor(0.0), (), {}),
        (canberra_distance, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.tensor(1.1), (), {}),
        (canberra_distance, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor(0.0), (), {}),
        (canberra_distance, torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]), torch.tensor(2.0), (), {}),
        (canberra_distance, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 6.0, 8.0]), torch.tensor(1.5545454545454546), (), {}),

        (minkowski_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.exp(-torch.tensor(0.0)), (), {'p': 1}),
        (minkowski_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.exp(-torch.tensor(7.0)), (), {'p': 1}),
        (minkowski_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.exp(-torch.tensor(5.0)), (), {'p': 2}),
        (minkowski_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.exp(-torch.tensor(4.4979)), (), {'p': 3}),
        (minkowski_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.exp(-torch.tensor(4.28457229495)), (), {'p': 4}),
        (minkowski_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.exp(-torch.tensor(4.0)), (), {'p': torch.inf}),

        (manhattan_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.exp(-torch.tensor(0.0)), (), {}),
        (manhattan_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.exp(-torch.tensor(7.0)), (), {}),
        (manhattan_distance_kernel, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.exp(-torch.tensor(0.0)), (), {}),
        (manhattan_distance_kernel, torch.tensor([1.0, -2.0]), torch.tensor([-3.0, 4.0]), torch.exp(-torch.tensor(10.0)), (), {}),
        (manhattan_distance_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 6.0, 8.0]), torch.exp(-torch.tensor(12.0)), (), {}),

        (euclidean_distance_kernel, torch.tensor([0.0, 1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]), torch.exp(-torch.tensor(5.1961524227)),(), {}),
        (euclidean_distance_kernel, torch.tensor([1.0]), torch.tensor([4.0]), torch.exp(-torch.tensor(3.0)), (), {}),
        (euclidean_distance_kernel, torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, -1.0]), torch.exp(-torch.tensor(2.828427124)),(), {}),
        (euclidean_distance_kernel, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.exp(-torch.tensor(0.0)), (), {}),
        (euclidean_distance_kernel, torch.tensor([1.5, -2.7, 3.14]), torch.tensor([-1.5, 2.7, -3.14]), torch.exp(-torch.tensor(8.80899540243)), (), {}),

        (chebyshev_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.exp(-torch.tensor(0.0)), (), {}),
        (chebyshev_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.exp(-torch.tensor(4.0)), (), {}),
        (chebyshev_distance_kernel, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.exp(-torch.tensor(0.0)), (), {}),
        (chebyshev_distance_kernel, torch.tensor([1.0, -2.0]), torch.tensor([-3.0, 4.0]), torch.exp(-torch.tensor(6.0)), (), {}),
        (chebyshev_distance_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 6.0, 9.0]), torch.exp(-torch.tensor(6.0)), (), {}),

        (canberra_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.exp(-torch.tensor(0.0)), (), {}),
        (canberra_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([4.0, 6.0]), torch.exp(-torch.tensor(1.1)), (), {}),
        (canberra_distance_kernel, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.exp(-torch.tensor(0.0)), (), {}),
        (canberra_distance_kernel, torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]), torch.exp(-torch.tensor(2.0)), (), {}),
        (canberra_distance_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 6.0, 8.0]), torch.exp(-torch.tensor(1.5545454545454546)), (), {}),

        (exponential_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor(0.00033546), (), {'gamma': 1.0}),
        (exponential_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor(0.01831564), (), {'gamma': 0.5}),
        (exponential_kernel, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor(1.0), (), {'gamma': 1.0}),
        (exponential_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.exp(torch.tensor(-27.0)), (), {'gamma': 1.0}),

        (gaussian_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.exp(torch.tensor(-4)), (), {'sigma': 1.0}),
        (gaussian_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.exp(torch.tensor(-16)), (), {'sigma': 0.5}),
        (gaussian_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.exp(torch.tensor(-400)), (), {'sigma': 0.1}),
        (gaussian_rbf_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.exp(torch.tensor(-13.5)), (), {'sigma': 1.0}),
        (gaussian_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor(1.0), (), {'sigma': 1.0}),

        (laplacian_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.exp(torch.tensor(-4.0)), (), {'sigma': 1.0}),
        (laplacian_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.exp(torch.tensor(-2.0)), (), {'sigma': 2.0}),
        (laplacian_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.exp(torch.tensor(-8.0)), (), {'sigma': 0.5}),
        (laplacian_kernel, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.exp(torch.tensor(-9.0)), (), {'sigma': 1.0}),
        (laplacian_kernel, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor(torch.tensor(1.0)), (), {'sigma': 1.0}),

        (anisotropic_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.exp(torch.tensor(-8.0)), (), {'a_vector': torch.tensor([1.0, 1.0])}),
        (anisotropic_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.exp(torch.tensor(-8.0)), (), {'a_scalar': 1.0}),
        (anisotropic_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.exp(torch.tensor(-12.0)), (), {'a_vector': torch.tensor([1.0, 2.0])}),
        (anisotropic_rbf_kernel, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor(1.0), (), {'a_vector': torch.tensor([0.0, 0.0])}),

    ])
    def test_kernel_functions(self, kernel_func, x1, x2, expected, args, kwargs):
        if isinstance(expected, type) and issubclass(expected, Exception):
            print(kernel_func, x1, x2, expected, args, kwargs)
            with pytest.raises(expected):
                result = kernel_func(x1, x2, *args, **kwargs)
        else:
            result = kernel_func(x1, x2, *args, **kwargs)
            assert torch.allclose(result, expected, atol=1e-6)



import pytest
import torch
from tinybig.koala.linear_algebra import (
    batch_linear_kernel,
    batch_polynomial_kernel,
    batch_hyperbolic_tangent_kernel,
    batch_cosine_similarity_kernel,
    batch_minkowski_distance,
    batch_euclidean_distance,
    batch_chebyshev_distance,
    batch_canberra_distance,
    batch_exponential_kernel,
    batch_gaussian_rbf_kernel,
    batch_laplacian_kernel,
    batch_anisotropic_rbf_kernel,
    batch_custom_hybrid_kernel
)

@pytest.fixture
def sample_data_batch():
    # Update to a 3*5 tensor
    return torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                         [6.0, 7.0, 8.0, 9.0, 10.0],
                         [11.0, 12.0, 13.0, 14.0, 15.0]])


class Test_Batch_Kernel:
    def test_batch_linear_kernel(self, sample_data_batch):
        kernel_matrix = batch_linear_kernel(sample_data_batch)
        assert kernel_matrix.shape == (5, 5)

        # Calculate expected kernel matrix manually (dot product of columns)
        expected_matrix = torch.matmul(sample_data_batch.T, sample_data_batch)
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0,1], linear_kernel(x=sample_data_batch[:,0], x2=sample_data_batch[:,1]), atol=1e-6)

    def test_batch_polynomial_kernel(self, sample_data_batch):
        c, d = 1.0, 2
        kernel_matrix = batch_polynomial_kernel(sample_data_batch, c=c, d=d)
        assert kernel_matrix.shape == (5, 5)

        # Calculate expected polynomial kernel
        linear_kernel = torch.matmul(sample_data_batch.T, sample_data_batch)
        expected_matrix = (linear_kernel + c) ** d
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              polynomial_kernel(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1], c=c, d=d), atol=1e-6)

    def test_batch_hyperbolic_tangent_kernel(self, sample_data_batch):
        alpha, c = 1.0, 0.5
        kernel_matrix = batch_hyperbolic_tangent_kernel(sample_data_batch, alpha=alpha, c=c)
        assert kernel_matrix.shape == (5, 5)

        # Calculate expected hyperbolic tangent kernel
        linear_kernel = torch.matmul(sample_data_batch.T, sample_data_batch)
        expected_matrix = torch.tanh(alpha * linear_kernel + c)
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              hyperbolic_tangent_kernel(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1]), atol=1e-6)

    def test_batch_cosine_similarity_kernel(self, sample_data_batch):
        kernel_matrix = batch_cosine_similarity_kernel(sample_data_batch)
        assert kernel_matrix.shape == (5, 5)

        # Calculate expected cosine similarity matrix
        norms = torch.norm(sample_data_batch, dim=0)
        expected_matrix = torch.matmul(sample_data_batch.T, sample_data_batch) / (norms.unsqueeze(0) * norms.unsqueeze(1))
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              cosine_similarity_kernel(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1]), atol=1e-6)

    def test_batch_minkowski_distance(self, sample_data_batch):
        p = 2
        kernel_matrix = batch_minkowski_distance(sample_data_batch, p=p)
        assert kernel_matrix.shape == (5, 5)

        # Calculate expected Minkowski distance matrix (Euclidean for p=2)
        x_expanded_1 = sample_data_batch.unsqueeze(2)
        x_expanded_2 = sample_data_batch.unsqueeze(1)
        expected_matrix = torch.norm(x_expanded_1 - x_expanded_2, p=p, dim=0)
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              minkowski_distance(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1], p=p), atol=1e-6)

    def test_batch_euclidean_distance(self, sample_data_batch):
        kernel_matrix = batch_euclidean_distance(sample_data_batch)
        assert kernel_matrix.shape == (5, 5)

        # Expected Euclidean distance matrix (same as p=2 Minkowski)
        x_expanded_1 = sample_data_batch.unsqueeze(2)
        x_expanded_2 = sample_data_batch.unsqueeze(1)
        expected_matrix = torch.norm(x_expanded_1 - x_expanded_2, p=2, dim=0)
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              euclidean_distance(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1]), atol=1e-6)

    def test_batch_chebyshev_distance(self, sample_data_batch):
        kernel_matrix = batch_chebyshev_distance(sample_data_batch)
        assert kernel_matrix.shape == (5, 5)

        # Expected Chebyshev distance matrix (p = infinity)
        x_expanded_1 = sample_data_batch.unsqueeze(2)
        x_expanded_2 = sample_data_batch.unsqueeze(1)
        expected_matrix = torch.norm(x_expanded_1 - x_expanded_2, p=float('inf'), dim=0)
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              chebyshev_distance(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1]), atol=1e-6)

    def test_batch_canberra_distance(self, sample_data_batch):
        kernel_matrix = batch_canberra_distance(sample_data_batch)
        assert kernel_matrix.shape == (5, 5)

        # Calculate expected Canberra distance matrix
        x_expanded_1 = sample_data_batch.unsqueeze(2)
        x_expanded_2 = sample_data_batch.unsqueeze(1)
        numerator = torch.abs(x_expanded_1 - x_expanded_2)
        denominator = torch.abs(x_expanded_1) + torch.abs(x_expanded_2)
        expected_matrix = torch.sum(numerator / (denominator + 1e-10), dim=0)
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              canberra_distance(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1]), atol=1e-6)

    def test_batch_exponential_kernel(self, sample_data_batch):
        gamma = 0.5
        kernel_matrix = batch_exponential_kernel(sample_data_batch, gamma=gamma)
        assert kernel_matrix.shape == (5, 5)

        # Expected exponential kernel (exp(-gamma * Euclidean distance^2))
        x_expanded_1 = sample_data_batch.unsqueeze(2)
        x_expanded_2 = sample_data_batch.unsqueeze(1)
        distance_matrix = torch.norm(x_expanded_1 - x_expanded_2, p=2, dim=0)
        expected_matrix = torch.exp(-gamma * distance_matrix**2)
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              exponential_kernel(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1], gamma=gamma), atol=1e-6)

    def test_batch_gaussian_rbf_kernel(self, sample_data_batch):
        sigma = 1.0
        kernel_matrix = batch_gaussian_rbf_kernel(sample_data_batch, sigma=sigma)
        assert kernel_matrix.shape == (5, 5)

        # Expected Gaussian RBF kernel
        x_expanded_1 = sample_data_batch.unsqueeze(2)
        x_expanded_2 = sample_data_batch.unsqueeze(1)
        distance_matrix = torch.norm(x_expanded_1 - x_expanded_2, p=2, dim=0)
        expected_matrix = torch.exp(-distance_matrix**2 / (2 * sigma**2))
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              gaussian_rbf_kernel(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1]), atol=1e-6)

    def test_batch_laplacian_kernel(self, sample_data_batch):
        sigma = 1.0
        kernel_matrix = batch_laplacian_kernel(sample_data_batch, sigma=sigma)
        assert kernel_matrix.shape == (5, 5)

        # Expected Laplacian kernel (exp(-Manhattan distance / sigma))
        x_expanded_1 = sample_data_batch.unsqueeze(2)
        x_expanded_2 = sample_data_batch.unsqueeze(1)
        manhattan_distance = torch.norm(x_expanded_1 - x_expanded_2, p=1, dim=0)
        expected_matrix = torch.exp(-manhattan_distance / sigma)
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              laplacian_kernel(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1]), atol=1e-6)

    def test_batch_anisotropic_rbf_kernel(self, sample_data_batch):
        a_vector = torch.tensor([1.0, 2.0, 3.0])
        kernel_matrix = batch_anisotropic_rbf_kernel(sample_data_batch, a_vector=a_vector)
        assert kernel_matrix.shape == (5, 5)

        # Expected anisotropic RBF kernel (using a diagonal matrix from a_vector)
        diff_matrix = sample_data_batch.T[:, None, :] - sample_data_batch.T[None, :, :]
        A = torch.diag(a_vector)
        expected_matrix = torch.exp(-torch.einsum('ijk,kl,ijl->ij', diff_matrix, A, diff_matrix))
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              anisotropic_rbf_kernel(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1], a_vector=a_vector), atol=1e-6)

    def test_batch_custom_hybrid_kernel(self, sample_data_batch):
        kernels = [batch_linear_kernel, batch_cosine_similarity_kernel]
        kernel_matrix = batch_custom_hybrid_kernel(sample_data_batch, kernels=kernels, weights=[0.5, 0.5])
        assert kernel_matrix.shape == (5, 5)

        # Expected custom hybrid kernel as a weighted sum of linear and cosine similarity kernels
        linear = torch.matmul(sample_data_batch.T, sample_data_batch)
        norms = torch.norm(sample_data_batch, dim=0)
        cosine_similarity = torch.matmul(sample_data_batch.T, sample_data_batch) / (norms.unsqueeze(0) * norms.unsqueeze(1))
        expected_matrix = 0.5 * linear + 0.5 * cosine_similarity
        assert torch.allclose(kernel_matrix, expected_matrix, atol=1e-6)
        assert torch.allclose(kernel_matrix[0, 1],
                              custom_hybrid_kernel(x=sample_data_batch[:, 0], x2=sample_data_batch[:, 1], kernels=[linear_kernel, cosine_similarity_kernel], weights=[0.5, 0.5]), atol=1e-6)

