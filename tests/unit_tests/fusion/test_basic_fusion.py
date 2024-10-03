# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Test Base Fusion Class #
##########################

import pytest
import torch
from tinybig.fusion.basic_fusion import weighted_summation_fusion, summation_fusion, average_fusion, parameterized_weighted_summation_fusion  # Replace with actual path


@pytest.fixture
def weighted_summation_instance():
    weights = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32)  # Ensure float type
    return weighted_summation_fusion(dims=[2, 2, 2], weights=weights)


@pytest.fixture
def summation_instance():
    return summation_fusion(dims=[2, 2, 2])


@pytest.fixture
def average_instance():
    return average_fusion(dims=[2, 2, 2])


@pytest.fixture
def parameterized_weighted_instance():
    return parameterized_weighted_summation_fusion(dims=[2, 2, 2])


def test_weighted_summation_fusion(weighted_summation_instance):
    x = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
         torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
         torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)]

    result = weighted_summation_instance.forward(x)
    expected = (x[0] * 0.2 + x[1] * 0.5 + x[2] * 0.3)

    assert torch.allclose(result, expected, rtol=1e-5)


def test_summation_fusion(summation_instance):
    x = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
         torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
         torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)]

    result = summation_instance.forward(x)
    expected = x[0] + x[1] + x[2]

    assert torch.allclose(result, expected, rtol=1e-5)


def test_average_fusion(average_instance):
    x = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
         torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
         torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)]

    result = average_instance.forward(x)
    expected = (x[0] + x[1] + x[2]) / 3

    assert torch.allclose(result, expected, rtol=1e-5)


def test_parameterized_weighted_summation_fusion(parameterized_weighted_instance):
    x = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
         torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
         torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)]

    weights = torch.nn.Parameter(torch.tensor([0.3, 0.3, 0.4], dtype=torch.float32))  # Ensure float type
    result = parameterized_weighted_instance.forward(x, w=weights)
    expected = (x[0] * 0.3 + x[1] * 0.3 + x[2] * 0.4)

    assert torch.allclose(result, expected, rtol=1e-5)


def test_weighted_summation_fusion_invalid_weights(weighted_summation_instance):
    with pytest.raises(AssertionError):
        x = [torch.tensor([[1.0, 2.0]], dtype=torch.float32),
             torch.tensor([[5.0, 6.0]], dtype=torch.float32)]
        weighted_summation_instance.forward(x)  # Weights do not match input


def test_parameterized_weighted_summation_fusion_no_weights(parameterized_weighted_instance):
    x = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
         torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
         torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)]

    with pytest.raises(AssertionError):
        parameterized_weighted_instance.forward(x)  # No weights provided


if __name__ == '__main__':
    pytest.main()
