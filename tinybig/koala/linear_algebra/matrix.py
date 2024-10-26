# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################
# Matrix Operators #
####################

from typing import Union
import numpy as np
import scipy.sparse as sp
import torch


def matrix_power(mx: torch.Tensor, n: int) -> torch.Tensor:
    """
    The matrix power calculates the powers of input matrix.

    Parameters
    ----------
    mx: torch.Tensor
        The matrix to be powered.
    n: int
        The power of the matrix.

    Returns
    -------
    torch.Tensor
        The matrix power.
    """
    assert mx is not None and mx.ndim == 2

    if n == 0:
        # Return identity matrix of the same shape as mx
        return torch.eye(mx.shape[0], dtype=mx.dtype, device=mx.device)
    elif n == 1:
        return mx
    elif n % 2 == 0:
        half_power = matrix_power(mx, n // 2)
        return half_power @ half_power
    else:
        return mx @ matrix_power(mx, n - 1)


def accumulative_matrix_power(mx: torch.Tensor, n: int) -> torch.Tensor:
    """
    The accumulative matrix power is defined as the summation of matrix powers from 1 to n.

    Parameters
    ----------
    mx: torch.Tensor
        The input matrix.
    n: int
        The highest power order.

    Returns
    -------
    torch.Tensor
        The summation of matrix powers from 1 to n.
    """
    assert mx is not None and mx.ndim == 2

    ac_mx_power = mx.clone()  # Initialize with the first power (mx^1)
    adj_powers = mx.clone()   # Initialize with the first power (mx^1)

    for i in range(2, n + 1):
        adj_powers = mx @ adj_powers  # Compute the next power
        ac_mx_power += adj_powers     # Add to the accumulative sum

    return ac_mx_power


def degree_based_normalize_matrix(mx: torch.Tensor, mode: str = "row") -> torch.Tensor:
    """
    Degree-based normalization of the matrix.

    Parameters
    ----------
    mx: torch.Tensor
        The input matrix (can be dense or sparse).
    mode: str
        The normalization mode. Can be 'row', 'column', or 'row-column'.

    Returns
    -------
    torch.Tensor
        The normalized matrix.
    """
    sparse_tag = False
    if mx.is_sparse:
        # Convert sparse matrix to dense
        mx = mx.to_dense()
        sparse_tag = True

    assert mx is not None and mx.ndim == 2

    if mode == "row":
        # Row normalization: Divide each row by the square root of its row sum
        row_sums = mx.sum(dim=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_mx = mx / row_sums.unsqueeze(1)

    elif mode == "column":
        # Column normalization: Divide each column by the square root of its column sum
        col_sums = mx.sum(dim=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        normalized_mx = mx / col_sums.unsqueeze(0)

    elif mode == "row_column":
        # Step 1: Row normalization with sqrt
        row_sums = mx.sum(dim=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        mx = mx / torch.sqrt(row_sums.unsqueeze(1))

        # Step 2: Column normalization with sqrt
        col_sums = mx.sum(dim=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        normalized_mx = mx / torch.sqrt(col_sums.unsqueeze(0))

    else:
        raise ValueError("Invalid normalization option. Choose 'row', 'column', or 'row_column'.")

    if sparse_tag:
        normalized_mx = normalized_mx.to_sparse_coo()
    return normalized_mx


def operator_based_normalize_matrix(
    mx: torch.Tensor,
    mask_zeros: bool = False,
    rescale_factor: float = 1.0,
    operator: callable = torch.nn.functional.softmax,
    mode="row"
) -> torch.Tensor:
    """
    Applies normalization using a specified operator.

    Parameters
    ----------
    mx: torch.Tensor
        The input matrix (can be dense or sparse).
    rescale_factor: float
        Factor by which to rescale the input matrix.
    operator: callable
        Function to apply for normalization (e.g., softmax).
    mode: str
        The normalization mode. Can be 'row', 'column', or 'row-column'.

    Returns
    -------
    torch.Tensor
        The normalized matrix.
    """
    if mx.is_sparse:
        mx = mx.to_dense()  # Convert sparse matrix to dense

    assert mx is not None and mx.ndim == 2

    mx = mx * rescale_factor

    if mask_zeros:
        mask = (mx != 0).float()
        large_negative_value = -1e9
        masked_mx = mx.clone()
        masked_mx[mx == 0] = large_negative_value
    else:
        masked_mx = mx
        mask = None

    if mode == "row":
        normalized_mx = operator(masked_mx, dim=1)
    elif mode == 'column':
        normalized_mx = operator(masked_mx, dim=0)
    elif mode == "row-column":
        masked_mx = operator(masked_mx, dim=1)
        normalized_mx = operator(masked_mx, dim=0)
    else:
        raise ValueError("Invalid normalization option. Choose 'row', 'column', or 'row-column'.")

    if mask_zeros:
        normalized_mx = normalized_mx * mask

    return normalized_mx


def mean_std_based_normalize_matrix(mx: torch.Tensor, mode="column"):
    """
    Normalize the input tensor X based on the specified mode.

    Parameters:
        X (torch.Tensor): Input data tensor of shape (t, n) where t denotes timestamps and n denotes stock instances.
        mode (str): Mode of normalization. Possible values are "row", "column", or "row_column".

    Returns:
        torch.Tensor: Normalized tensor X.
    """
    if mode == "row":
        # Normalize each row (across instances)
        row_means = mx.mean(dim=1, keepdim=True)
        row_stds = mx.std(dim=1, keepdim=True)
        normalized_mx = (mx - row_means) / (row_stds + 1e-8)
    elif mode == "column":
        # Normalize each column (across time)
        col_means = mx.mean(dim=0, keepdim=True)
        col_stds = mx.std(dim=0, keepdim=True)
        normalized_mx = (mx - col_means) / (col_stds + 1e-8)
    elif mode in ["row_column", "column_row", "all"]:
        global_mean = mx.mean()
        global_std = mx.std()
        normalized_mx = (mx - global_mean) / (global_std + 1e-8)
    else:
        raise ValueError("Invalid mode. Choose from 'row', 'column', or 'row_column'.")
    return normalized_mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


