# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################################
# Test Matrix Operators in koala.linear_algebra #
#################################################

# test_matrix_operations.py
import pytest
import torch
import scipy.sparse as sp
from tinybig.koala.linear_algebra import (
    matrix_power,
    accumulative_matrix_power,
    sparse_matrix_to_torch_sparse_tensor,
    degree_based_normalize_matrix,
    operator_based_normalize_matrix,
)

@pytest.fixture
def dense_matrix():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])

@pytest.fixture
def sparse_matrix():
    return sp.csr_matrix([[1.0, 2.0], [0.0, 3.0]])


class Test_Matrix_Operators:
    def test_matrix_power_2(self, dense_matrix):
        assert torch.allclose(matrix_power(dense_matrix, 0), torch.eye(2))
        assert torch.allclose(matrix_power(dense_matrix, 1), dense_matrix)
        expected_power_2 = torch.tensor([[7.0, 10.0], [15.0, 22.0]])
        assert torch.allclose(matrix_power(dense_matrix, 2), expected_power_2)

    def test_matrix_power_4(self, dense_matrix):
        assert torch.allclose(matrix_power(dense_matrix, 0), torch.eye(2))
        assert torch.allclose(matrix_power(dense_matrix, 1), dense_matrix)
        expected_power_2 = torch.tensor([[199.0, 290.0], [435.0, 634.0]])
        assert torch.allclose(matrix_power(dense_matrix, 4), expected_power_2)

    def test_accumulative_matrix_power(self, dense_matrix):
        expected_accum = dense_matrix + torch.tensor([[7.0, 10.0], [15.0, 22.0]])
        assert torch.allclose(accumulative_matrix_power(dense_matrix, 2), expected_accum)

    def test_sparse_matrix_to_torch_sparse_tensor(self, sparse_matrix):
        torch_sparse = sparse_matrix_to_torch_sparse_tensor(sparse_matrix, dtype=torch.float32)
        expected_indices = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)
        expected_values = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(torch_sparse._indices(), expected_indices)
        assert torch.equal(torch_sparse._values(), expected_values)

    def test_dense_matrix_to_torch_sparse_tensor(self, dense_matrix):
        torch_dense = sparse_matrix_to_torch_sparse_tensor(dense_matrix, dtype=torch.float32)
        expected_matrix = dense_matrix
        assert torch.equal(torch_dense, expected_matrix)

    def test_degree_based_normalize_matrix_row(self, dense_matrix):
        normalized = degree_based_normalize_matrix(dense_matrix, mode="row")
        row_sums = dense_matrix.sum(dim=1)
        expected = dense_matrix / row_sums.unsqueeze(1)
        assert torch.allclose(normalized, expected)

    def test_degree_based_normalize_matrix_column(self, dense_matrix):
        normalized = degree_based_normalize_matrix(dense_matrix, mode="column")
        col_sums = dense_matrix.sum(dim=0)
        expected = dense_matrix / col_sums.unsqueeze(0)
        assert torch.allclose(normalized, expected)

    def test_degree_based_normalize_matrix_row_column(self, dense_matrix):
        normalized = degree_based_normalize_matrix(dense_matrix, mode="row_column")
        row_sums = dense_matrix.sum(dim=1).sqrt()
        row_normalized = dense_matrix / row_sums.unsqueeze(1)
        col_sums = row_normalized.sum(dim=0).sqrt()
        expected = row_normalized / col_sums.unsqueeze(0)
        assert torch.allclose(normalized, expected)

    def test_operator_based_normalize_matrix(self, dense_matrix):
        normalized = operator_based_normalize_matrix(dense_matrix, rescale_factor=1.0, operator=torch.nn.functional.softmax, mode="row")
        softmax = torch.nn.functional.softmax(dense_matrix, dim=1)
        assert torch.allclose(normalized, softmax)

    @pytest.mark.parametrize("mode", ["row", "column", "row-column"])
    def test_operator_based_normalize_matrix_modes(self, dense_matrix, mode):
        normalized = operator_based_normalize_matrix(dense_matrix, rescale_factor=1.0, operator=torch.nn.functional.softmax, mode=mode)
        if mode == "row":
            assert torch.allclose(normalized, torch.nn.functional.softmax(dense_matrix, dim=1))
        elif mode == "column":
            assert torch.allclose(normalized, torch.nn.functional.softmax(dense_matrix, dim=0))
        elif mode == "row-column":
            row_softmax = torch.nn.functional.softmax(dense_matrix, dim=1)
            col_softmax = torch.nn.functional.softmax(row_softmax, dim=0)
            assert torch.allclose(normalized, col_softmax)



