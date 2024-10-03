# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################################
# Test Metrics in koala.statistics #
####################################


# test_statistical_metrics.py

import pytest
import torch
from tinybig.koala.statistics.metrics import metric

@pytest.fixture
def tensor_1d():
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

@pytest.fixture
def tensor_2d():
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

@pytest.fixture
def weights_1d():
    return torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])

@pytest.fixture
def weights_2d():
    return torch.tensor([0.2, 0.3, 0.5])

def test_mean(tensor_1d):
    result = metric('mean', tensor_1d)
    expected = torch.mean(tensor_1d)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_batch_mean(tensor_2d):
    result = metric('batch_mean', tensor_2d, dim=1)
    expected = torch.mean(tensor_2d, dim=1)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_weighted_mean(tensor_1d, weights_1d):
    result = metric('weighted_mean', tensor_1d, weights=weights_1d)
    expected = torch.sum(tensor_1d * weights_1d) / torch.sum(weights_1d)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_batch_weighted_mean(tensor_2d, weights_2d):
    result = metric('batch_weighted_mean', tensor_2d, weights=weights_2d, dim=1)
    expected = torch.sum(tensor_2d * weights_2d.unsqueeze(0), dim=1) / torch.sum(weights_2d)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_geometric_mean(tensor_1d):
    result = metric('geometric_mean', tensor_1d)
    expected = torch.exp(torch.mean(torch.log(tensor_1d)))
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_harmonic_mean(tensor_1d):
    result = metric('harmonic_mean', tensor_1d)
    expected = tensor_1d.numel() / torch.sum(1 / tensor_1d)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_median(tensor_1d):
    result = metric('median', tensor_1d)
    expected = torch.median(tensor_1d)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_mode(tensor_1d):
    result, _ = metric('mode', tensor_1d)
    expected, _ = torch.mode(tensor_1d)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_entropy():
    prob_dist = torch.tensor([0.2, 0.3, 0.5])
    result = metric('entropy', prob_dist)
    expected = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_variance(tensor_1d):
    result = metric('variance', tensor_1d)
    expected = torch.var(tensor_1d)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_standard_deviation(tensor_1d):
    result = metric('std', tensor_1d)
    expected = torch.std(tensor_1d)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_skewness(tensor_1d):
    result = metric('skewness', tensor_1d)
    expected = torch.mean((tensor_1d - torch.mean(tensor_1d)) ** 3) / (torch.std(tensor_1d, unbiased=False) ** 3 + 1e-12)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

# Batch versions of the tests

def test_batch_geometric_mean(tensor_2d):
    result = metric('batch_geometric_mean', tensor_2d, dim=1)
    expected = torch.exp(torch.mean(torch.log(tensor_2d), dim=1))
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_batch_harmonic_mean(tensor_2d):
    result = metric('batch_harmonic_mean', tensor_2d, dim=1)
    reciprocal_sum = torch.sum(1 / tensor_2d, dim=1)
    expected = tensor_2d.shape[1] / reciprocal_sum
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_batch_median(tensor_2d):
    result = metric('batch_median', tensor_2d, dim=1)
    expected = torch.median(tensor_2d, dim=1).values
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_batch_mode(tensor_2d):
    result, _ = metric('batch_mode', tensor_2d, dim=1)
    expected, _ = torch.mode(tensor_2d, dim=1)
    assert torch.all(result == expected), f"Expected {expected}, but got {result}"

def test_batch_variance(tensor_2d):
    result = metric('batch_variance', tensor_2d, dim=1)
    expected = torch.var(tensor_2d, dim=1)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_batch_standard_deviation(tensor_2d):
    result = metric('batch_std', tensor_2d, dim=1)
    expected = torch.std(tensor_2d, dim=1)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_batch_skewness(tensor_2d):
    result = metric('batch_skewness', tensor_2d, dim=1)
    expected = torch.mean((tensor_2d - torch.mean(tensor_2d, dim=1, keepdim=True)) ** 3, dim=1) / \
               (torch.std(tensor_2d, dim=1, unbiased=False, keepdim=True) ** 3 + 1e-12)
    assert torch.allclose(result, expected.squeeze()), f"Expected {expected}, but got {result}"
