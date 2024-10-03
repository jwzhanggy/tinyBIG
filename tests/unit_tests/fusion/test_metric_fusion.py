# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Test Metric based Fusion Class #
##################################


import pytest
import torch
from tinybig.fusion.metric_fusion import (
    max_fusion,
    min_fusion,
    sum_fusion,
    mean_fusion,
    prod_fusion,
    median_fusion,
    l1_norm_fusion,
    l2_norm_fusion,
)  # Replace with actual path


@pytest.fixture
def max_fusion_instance():
    return max_fusion(dims=[2, 2, 2])


@pytest.fixture
def min_fusion_instance():
    return min_fusion(dims=[2, 2, 2])


@pytest.fixture
def sum_fusion_instance():
    return sum_fusion(dims=[2, 2, 2])


@pytest.fixture
def mean_fusion_instance():
    return mean_fusion(dims=[2, 2, 2])


@pytest.fixture
def prod_fusion_instance():
    return prod_fusion(dims=[2, 2, 2])


@pytest.fixture
def median_fusion_instance():
    return median_fusion(dims=[2, 2, 2])


@pytest.fixture
def l1_norm_fusion_instance():
    return l1_norm_fusion(dims=[2, 2, 2])


@pytest.fixture
def l2_norm_fusion_instance():
    return l2_norm_fusion(dims=[2, 2, 2])


def test_max_fusion(max_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 0.0], [7.0, 1.0]], dtype=torch.float32),
        torch.tensor([[9.0, 2.0], [11.0, 3.0]], dtype=torch.float32)
    ]

    result = max_fusion_instance.forward(x)
    expected = torch.tensor([[9.0, 2.0], [11.0, 4.0]], dtype=torch.float32)

    assert torch.allclose(result, expected, rtol=1e-5)


def test_min_fusion(min_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 0.0], [7.0, 1.0]], dtype=torch.float32),
        torch.tensor([[9.0, 2.0], [11.0, 3.0]], dtype=torch.float32)
    ]

    result = min_fusion_instance.forward(x)
    expected = torch.tensor([[1.0, 0.0], [3.0, 1.0]], dtype=torch.float32)

    assert torch.allclose(result, expected, rtol=1e-5)


def test_sum_fusion(sum_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    result = sum_fusion_instance.forward(x)
    expected = torch.tensor([[15.0, 18.0], [21.0, 24.0]], dtype=torch.float32)

    assert torch.allclose(result, expected, rtol=1e-5)


def test_mean_fusion(mean_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    result = mean_fusion_instance.forward(x)
    expected = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

    assert torch.allclose(result, expected, rtol=1e-5)


def test_prod_fusion(prod_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    result = prod_fusion_instance.forward(x)
    expected = torch.tensor([[45.0, 120.0], [231.0, 384.0]], dtype=torch.float32)

    assert torch.allclose(result, expected, rtol=1e-5)


def test_median_fusion(median_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    result = median_fusion_instance.forward(x)
    expected = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

    assert torch.allclose(result, expected, rtol=1e-5)


def test_l1_norm_fusion(l1_norm_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    result = l1_norm_fusion_instance.forward(x)
    expected = torch.tensor([[15.0, 18.0], [21.0, 24.0]], dtype=torch.float32)  # Should match sum as L1 norm

    assert torch.allclose(result, expected, rtol=1e-5)


def test_l2_norm_fusion(l2_norm_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    result = l2_norm_fusion_instance.forward(x)
    expected = torch.tensor([[10.3441, 11.8322], [13.3791, 14.9666]], dtype=torch.float32)  # As an example

    assert torch.allclose(result, expected, rtol=1e-5)


if __name__ == '__main__':
    pytest.main()

