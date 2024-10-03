# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

########################################
# Test Metrics in koala.linear_algebra #
########################################


import pytest
import torch

from tinybig.koala.linear_algebra.metric import norm, batch_norm, l1_norm, batch_l1_norm, l2_norm, batch_l2_norm, sum, batch_sum, prod, batch_prod, max, batch_max, min, batch_min

# Define test data
@pytest.fixture
def test_1d_tensor():
    return torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], dtype=torch.float32)

@pytest.fixture
def test_2d_tensor():
    return torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=torch.float32)


class Test_Metrics:
    @pytest.mark.parametrize("p", [1, 2, 'fro', float('inf'), -float('inf')])
    def test_norm(self, test_1d_tensor, p):
        result = norm(test_1d_tensor, p)
        expected = torch.norm(test_1d_tensor, p=p)
        assert torch.allclose(result, expected), f"norm failed for p={p}"

    def test_norm_value_error_for_nuc_norm(self, test_1d_tensor):
        with pytest.raises(ValueError, match="the nuc-norm cannot be applied to 1d tensor inputs..."):
            norm(test_1d_tensor, p='nuc')

    @pytest.mark.parametrize("p", [1, 2, 'fro', 'nuc', float('inf'), -float('inf')])
    @pytest.mark.parametrize("dim", [0, 1])
    def test_batch_norm(self, test_2d_tensor, p, dim):
        result = batch_norm(test_2d_tensor, p, dim)
        if p == 'nuc':
            expected = torch.norm(test_2d_tensor, p=p)
        else:
            expected = torch.norm(test_2d_tensor, p=p, dim=dim)
        assert torch.allclose(result, expected), f"batch_norm failed for p={p}, dim={dim}"

    def test_l1_norm(self, test_1d_tensor):
        result = l1_norm(test_1d_tensor)
        expected = torch.norm(test_1d_tensor, p=1)
        assert torch.allclose(result, expected), "l1_norm failed"

    @pytest.mark.parametrize("dim", [0, 1])
    def test_batch_l1_norm(self, test_2d_tensor, dim):
        result = batch_l1_norm(test_2d_tensor, dim)
        expected = torch.norm(test_2d_tensor, p=1, dim=dim)
        assert torch.allclose(result, expected), f"batch_l1_norm failed for dim={dim}"

    def test_l2_norm(self, test_1d_tensor):
        result = l2_norm(test_1d_tensor)
        expected = torch.norm(test_1d_tensor, p=2)
        assert torch.allclose(result, expected), "l2_norm failed"

    @pytest.mark.parametrize("dim", [0, 1])
    def test_batch_l2_norm(self, test_2d_tensor, dim):
        result = batch_l2_norm(test_2d_tensor, dim)
        expected = torch.norm(test_2d_tensor, p=2, dim=dim)
        assert torch.allclose(result, expected), f"batch_l2_norm failed for dim={dim}"

    def test_sum(self, test_1d_tensor):
        result = sum(test_1d_tensor)
        expected = torch.sum(test_1d_tensor)
        assert torch.allclose(result, expected), "sum failed"

    @pytest.mark.parametrize("dim", [0, 1])
    def test_batch_sum(self, test_2d_tensor, dim):
        result = batch_sum(test_2d_tensor, dim)
        expected = torch.sum(test_2d_tensor, dim=dim)
        assert torch.allclose(result, expected), f"batch_sum failed for dim={dim}"

    def test_prod(self, test_1d_tensor):
        result = prod(test_1d_tensor)
        expected = torch.prod(test_1d_tensor)
        assert torch.allclose(result, expected), "prod failed"

    @pytest.mark.parametrize("dim", [0, 1])
    def test_batch_prod(self, test_2d_tensor, dim):
        result = batch_prod(test_2d_tensor, dim)
        expected = torch.prod(test_2d_tensor, dim=dim)
        assert torch.allclose(result, expected), f"batch_prod failed for dim={dim}"

    def test_max(self, test_1d_tensor):
        result = max(test_1d_tensor)
        expected = torch.max(test_1d_tensor)
        assert torch.allclose(result, expected), "max failed"

    @pytest.mark.parametrize("dim", [0, 1])
    def test_batch_max(self, test_2d_tensor, dim):
        result = batch_max(test_2d_tensor, dim)
        expected = torch.max(test_2d_tensor, dim=dim)[0]
        assert torch.allclose(result, expected), f"batch_max failed for dim={dim}"

    def test_min(self, test_1d_tensor):
        result = min(test_1d_tensor)
        expected = torch.min(test_1d_tensor)
        assert torch.allclose(result, expected), "min failed"

    @pytest.mark.parametrize("dim", [0, 1])
    def test_batch_min(self, test_2d_tensor, dim):
        result = batch_min(test_2d_tensor, dim)
        expected = torch.min(test_2d_tensor, dim=dim)[0]
        assert torch.allclose(result, expected), f"batch_min failed for dim={dim}"


