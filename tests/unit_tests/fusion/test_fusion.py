# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Test Base Fusion Class #
##########################

import pytest
import torch
from tinybig.module.base_fusion import fusion


class TestFusion(fusion):
    def calculate_n(self, dims=None, *args, **kwargs):
        return sum(dims) if dims else 0

    def calculate_l(self, *args, **kwargs):
        return len(args)  # Just a dummy implementation for testing

    def forward(self, x, w=None, device='cpu', *args, **kwargs):
        return torch.cat(x, dim=0) if isinstance(x, list) else x


@pytest.fixture
def fusion_instance():
    return TestFusion(dims=[1, 2, 3], name='test_fusion')


def test_initialization(fusion_instance):
    assert fusion_instance.get_name() == 'test_fusion'
    assert fusion_instance.get_dims() == [1, 2, 3]
    assert fusion_instance.get_num() == 3


def test_get_dim(fusion_instance):
    assert fusion_instance.get_dim(0) == 1
    assert fusion_instance.get_dim(1) == 2
    assert fusion_instance.get_dim(2) == 3
    with pytest.raises(IndexError):
        fusion_instance.get_dim(3)  # Out of bounds


def test_pre_process(fusion_instance):
    sample_tensor = torch.tensor([[1, 2], [3, 4]])
    # Assuming a simple preprocess function for demonstration
    fusion_instance.preprocess_functions = [lambda x: x + 1]
    processed_tensor = fusion_instance.pre_process(sample_tensor)
    assert torch.equal(processed_tensor, torch.tensor([[2, 3], [4, 5]]))


def test_post_process(fusion_instance):
    sample_tensor = torch.tensor([[1, 2], [3, 4]])
    # Assuming a simple postprocess function for demonstration
    fusion_instance.postprocess_functions = [lambda x: x * 2]
    processed_tensor = fusion_instance.post_process(sample_tensor)
    assert torch.equal(processed_tensor, torch.tensor([[2, 4], [6, 8]]))


def test_calculate_n(fusion_instance):
    assert fusion_instance.calculate_n([1, 2, 3]) == 6
    assert fusion_instance.calculate_n() == 0


def test_calculate_l(fusion_instance):
    assert fusion_instance.calculate_l(1, 2, 3) == 3  # Number of arguments


def test_forward(fusion_instance):
    sample_tensors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    result = fusion_instance.forward(sample_tensors)
    assert torch.equal(result, torch.tensor([1, 2, 3, 4]))


def test_forward_single_tensor(fusion_instance):
    sample_tensor = torch.tensor([1, 2, 3])
    result = fusion_instance.forward(sample_tensor)
    assert torch.equal(result, sample_tensor)  # Should return the same tensor


if __name__ == '__main__':
    pytest.main()
