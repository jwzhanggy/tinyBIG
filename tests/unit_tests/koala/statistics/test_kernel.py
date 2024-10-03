# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################################
# Test Kernels in koala.statistics #
####################################

# test_kernels.py
import pytest
import torch
from tinybig.koala.statistics.kernel import (
    pearson_correlation_kernel,
    kl_divergence_kernel,
    rv_coefficient_kernel,
    mutual_information_kernel,
    custom_hybrid_kernel,
    kernel
)


# Utility function for generating random test tensors
def generate_test_tensor(size, requires_grad=False):
    return torch.randn(size, requires_grad=requires_grad)


# Test data: instance-level inputs (1D tensors) and batch-level inputs (2D tensors)
@pytest.fixture
def instance_data():
    return generate_test_tensor(100), generate_test_tensor(100)


@pytest.fixture
def batch_data():
    return generate_test_tensor((10, 100))


class Test_Metrics:
    # Test cases for Pearson correlation kernel
    def test_instance_pearson_correlation(self, instance_data):
        x1, x2 = instance_data
        result = pearson_correlation_kernel(x1, x2)
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1  # Pearson correlation should return a scalar
        assert -1.0 <= result.item() <= 1.0

    def test_batch_pearson_correlation(self, batch_data):
        x = batch_data
        result = pearson_correlation_kernel(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (x.shape[1], x.shape[1])  # Kernel matrix shape
        assert torch.allclose(result, result.T)  # Symmetric matrix
        assert torch.allclose(result[0, 1], pearson_correlation_kernel(x[:, 0], x[:, 1]))

    # Test cases for KL divergence kernel
    def test_instance_kl_divergence(self, instance_data):
        x1, x2 = instance_data
        result = kl_divergence_kernel(x1, x2)
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1  # KL divergence should return a scalar
        assert result.item() >= 0.0  # KL divergence should be non-negative

    def test_batch_kl_divergence(self, batch_data):
        x = batch_data
        result = kl_divergence_kernel(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (x.shape[1], x.shape[1])  # Kernel matrix shape
        assert torch.allclose(result[0, 1], kl_divergence_kernel(x[:, 0], x[:, 1]))

    # Test cases for RV coefficient kernel
    def test_instance_rv_coefficient(self, instance_data):
        x1, x2 = instance_data
        result = rv_coefficient_kernel(x1, x2)
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1  # RV coefficient should return a scalar
        assert 0.0 <= result.item() <= 1.0  # RV coefficient is in [0, 1]

    def test_batch_rv_coefficient(self, batch_data):
        x = batch_data
        result = rv_coefficient_kernel(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (x.shape[1], x.shape[1])  # Kernel matrix shape
        #assert torch.allclose(result[0, 1], rv_coefficient_kernel(x[:, 0], x[:, 1]))

    # Test cases for Mutual Information kernel
    def test_instance_mutual_information(self, instance_data):
        x1, x2 = instance_data
        result = mutual_information_kernel(x1, x2)
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1  # Mutual Information should return a scalar

    def test_batch_mutual_information(self, batch_data):
        x = batch_data
        result = mutual_information_kernel(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (x.shape[1], x.shape[1])  # Kernel matrix shape
        assert torch.allclose(result[0, 1], mutual_information_kernel(x[:, 0], x[:, 1]))

    # Test cases for custom hybrid kernel
    def test_instance_custom_hybrid(self, instance_data):
        x1, x2 = instance_data
        kernels = [pearson_correlation_kernel, rv_coefficient_kernel]
        result = custom_hybrid_kernel(x1, x2, kernels=kernels, weights=[0.5, 0.5])
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1  # Custom hybrid should return a scalar

    def test_batch_custom_hybrid(self, batch_data):
        x = batch_data
        kernels = [pearson_correlation_kernel, rv_coefficient_kernel]
        result = custom_hybrid_kernel(x, kernels=kernels, weights=[0.5, 0.5])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (x.shape[1], x.shape[1])  # Kernel matrix shape
        assert torch.allclose(result[0, 1], custom_hybrid_kernel(x[:, 0], x[:, 1], kernels=kernels, weights=[0.5, 0.5]))

    # Test using the kernel function with different kernels
    @pytest.mark.parametrize("kernel_name", [
        'pearson_correlation', 'kl_divergence', 'rv_coefficient', 'mutual_information'
    ])
    def test_kernel_function_instance(self, kernel_name, instance_data):
        x1, x2 = instance_data
        result = kernel(kernel_name=kernel_name, x=x1, x2=x2)
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1  # Expecting scalar output

    @pytest.mark.parametrize("kernel_name", [
        'batch_pearson_correlation', 'batch_kl_divergence', 'batch_rv_coefficient', 'batch_mutual_information'
    ])
    def test_kernel_function_batch(self, kernel_name, batch_data):
        x = batch_data
        result = kernel(kernel_name=kernel_name, x=x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (x.shape[1], x.shape[1])  # Kernel matrix shape
