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


def test_initialization(concatenation_instance):
    assert concatenation_instance.get_name() == "concatenation_fusion"
    assert concatenation_instance.get_dims() == [2, 2, 2]
    assert concatenation_instance.get_num() == 3


def test_calculate_n(concatenation_instance):
    assert concatenation_instance.calculate_n([2, 2, 2]) == 6
    assert concatenation_instance.calculate_n() == 6


def test_calculate_l(concatenation_instance):
    assert concatenation_instance.calculate_l() == 0


def test_forward_with_list(concatenation_instance):
    x = [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
         torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
         torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)]

    result = concatenation_instance.forward(x)
    expected = torch.tensor([[1.0, 2.0, 5.0, 6.0, 9.0, 10.0],
                             [3.0, 4.0, 7.0, 8.0, 11.0, 12.0]], dtype=torch.float32)
    assert torch.equal(result, expected)
    assert torch.allclose(result, expected, rtol=1e-5)


def test_forward_with_tensor(concatenation_instance):
    x = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32)
    ]

    result = concatenation_instance.forward(x)
    expected = torch.tensor([[1.0, 2.0, 5.0, 6.0, 9.0, 10.0],
                             [3.0, 4.0, 7.0, 8.0, 11.0, 12.0]], dtype=torch.float32)

    assert torch.equal(result, expected)
    assert torch.allclose(result, expected, rtol=1e-5)


def test_forward_with_empty_list(concatenation_instance):
    x = []
    with pytest.raises(ValueError):
        concatenation_instance.forward(x)  # Expecting a ValueError due to empty input


def test_forward_with_different_sizes(concatenation_instance):
    x = [torch.tensor([[1.0]], dtype=torch.float32),
         torch.tensor([[2.0], [3.0]], dtype=torch.float32),
         torch.tensor([[4.0]], dtype=torch.float32)]

    with pytest.raises(ValueError):
        result = concatenation_instance.forward(x)


def test_post_process_called(concatenation_instance):
    # Mock the post_process method to confirm it is called
    def mock_post_process(x, device='cpu', *args, **kwargs):
        return x * 2  # Just an example operation

    concatenation_instance.post_process = mock_post_process
    x = [torch.tensor([[1.0, 2.0]], dtype=torch.float32),
         torch.tensor([[3.0, 4.0]], dtype=torch.float32)]

    result = concatenation_instance.forward(x)
    expected = torch.tensor([[2.0, 4.0, 6.0, 8.0]], dtype=torch.float32)  # Result after mock post-processing

    assert torch.equal(result, expected)
    assert torch.allclose(result, expected, rtol=1e-5)


if __name__ == '__main__':
    pytest.main()
