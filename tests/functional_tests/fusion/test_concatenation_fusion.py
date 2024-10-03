# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#########################################
# Test Concatenation based Fusion Class #
#########################################


import pytest
import torch
from tinybig.fusion.concatenation_fusion import concatenation_fusion  # Replace with actual path


@pytest.fixture
def concatenation_instance():
    return concatenation_fusion(dims=[2, 2, 2])  # Example dimensions


def test_forward_with_tensor(concatenation_instance):
    x = [
        torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0, 9.0, 10.0], [11.0, 12.0, 11.0, 12.0]], dtype=torch.float32)
    ]

    result = concatenation_instance.forward(x)
    expected = torch.tensor([[1.,  2.,  3.,  5.,  6.,  9., 10.,  9., 10.],
                             [3.,  4.,  5.,  7.,  8., 11., 12., 11., 12.]], dtype=torch.float32)

    assert torch.equal(result, expected)
    assert torch.allclose(result, expected, rtol=1e-5)