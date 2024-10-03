# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################################################
# Test Parameterized Concatenation based Fusion Class #
#######################################################

import pytest
import torch
from tinybig.fusion.parameterized_concatenation_fusion import (
    parameterized_concatenation_fusion,
    lowrank_parameterized_concatenation_fusion,
    hm_parameterized_concatenation_fusion,
    lphm_parameterized_concatenation_fusion,
    dual_lphm_parameterized_concatenation_fusion,
    random_matrix_adaption_parameterized_concatenation_fusion,
)  # Replace with actual path


@pytest.fixture
def parameterized_fusion_instance():
    return parameterized_concatenation_fusion(n=3, dims=[2, 2, 2])


def test_parameterized_fusion_initialization(parameterized_fusion_instance):
    assert parameterized_fusion_instance.get_name() == "parameterized_concatenation_fusion"
    assert parameterized_fusion_instance.n == 3
    assert parameterized_fusion_instance.dims == [2, 2, 2]


def test_forward_with_valid_input(parameterized_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    # Calculate the lengths for w based on the last dimension sizes
    last_dims = [t.shape[-1] for t in x]
    w_length = parameterized_fusion_instance.calculate_l(last_dims)

    w = torch.randn(1, w_length)

    result = parameterized_fusion_instance.forward(x, w=w)

    assert result.shape == (2, 3)  # Check expected output shape


def test_forward_empty_input(parameterized_fusion_instance):
    with pytest.raises(ValueError):
        parameterized_fusion_instance.forward([])  # Expecting a ValueError


def test_forward_with_different_shapes(parameterized_fusion_instance):
    x = [
        torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    ]
    with pytest.raises(ValueError):
        parameterized_fusion_instance.forward(x)  # Expecting a ValueError


def test_lowrank_parameterized_fusion():
    fusion_instance = lowrank_parameterized_concatenation_fusion(n=3, r=2, dims=[2, 2, 2])
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    last_dims = [t.shape[-1] for t in x]
    w_length = fusion_instance.calculate_l(last_dims)

    w = torch.randn(1, w_length)

    result = fusion_instance.forward(x, w=w)

    assert result.shape == (2, 3)  # Check expected output shape


def test_hm_parameterized_fusion():
    fusion_instance = hm_parameterized_concatenation_fusion(n=3, p=3, q=2, dims=[2, 2, 2])
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    last_dims = [t.shape[-1] for t in x]
    w_length = fusion_instance.calculate_l(last_dims)

    w = torch.randn(1, w_length)

    result = fusion_instance.forward(x, w=w)

    assert result.shape == (2, 3)  # Check expected output shape


def test_lphm_parameterized_fusion():
    fusion_instance = lphm_parameterized_concatenation_fusion(n=3, r=2, p=3, q=2, dims=[2, 2, 2])
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    last_dims = [t.shape[-1] for t in x]
    w_length = fusion_instance.calculate_l(last_dims)

    w = torch.randn(1, w_length)

    result = fusion_instance.forward(x, w=w)

    assert result.shape == (2, 3)  # Check expected output shape


def test_dual_lphm_parameterized_fusion():
    fusion_instance = dual_lphm_parameterized_concatenation_fusion(n=3, r=2, p=3, q=2, dims=[2, 2, 2])
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    last_dims = [t.shape[-1] for t in x]
    w_length = fusion_instance.calculate_l(last_dims)

    w = torch.randn(1, w_length)

    result = fusion_instance.forward(x, w=w)

    assert result.shape == (2, 3)  # Check expected output shape


def test_random_matrix_adaption_parameterized_fusion():
    fusion_instance = random_matrix_adaption_parameterized_concatenation_fusion(n=3, r=2, dims=[2, 2, 2])
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    last_dims = [t.shape[-1] for t in x]
    w_length = fusion_instance.calculate_l(last_dims)

    w = torch.randn(1, w_length)

    result = fusion_instance.forward(x, w=w)

    assert result.shape == (2, 3)  # Check expected output shape


if __name__ == '__main__':
    pytest.main()
