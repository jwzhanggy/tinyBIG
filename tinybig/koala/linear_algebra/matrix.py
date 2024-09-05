import scipy.sparse as sp
import torch
import numpy as np


def matrix_power(mx, n):
    if n == 0:
        return sp.eye(mx.shape[0], format='csr')
    elif n == 1:
        return mx
    elif n % 2 == 0:
        half_power = matrix_power(mx, n // 2)
        return half_power @ half_power
    else:
        return mx @ matrix_power(mx, n - 1)


def accumulative_matrix_power(mx, n):
    ac_mx_power = mx
    adj_powers = mx
    for i in range(2, n + 1):
        adj_powers = mx @ adj_powers
        ac_mx_power += adj_powers
    return ac_mx_power


def sparse_matrix_to_torch_sparse_tensor(sparse_mx, dtype=np.float32):
    sparse_mx = sparse_mx.tocoo().astype(dtype)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def normalize_matrix(mx, mode="row-column"):
    """
    Normalize a matrix mx using its degree matrix. The normalization is based on
    either row sums, column sums, or both, where the normalization divides by
    the square root of the row or column sums.

    Parameters:
    mx (np.ndarray or scipy.sparse matrix): Input matrix to be normalized
    normalization (str): One of "row", "column", or "row-column" for the normalization type

    Returns:
    np.ndarray or scipy.sparse matrix: Normalized matrix
    """
    if sp.issparse(mx):
        mx = mx.toarray()  # Convert to dense if it's sparse, you can handle sparse normalization separately if needed

    if mode == "row":
        # Row normalization: Divide each row by the square root of its row sum
        row_sums = mx.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_mx = mx / np.sqrt(row_sums[:, np.newaxis])

    elif mode == "column":
        # Column normalization: Divide each column by the square root of its column sum
        col_sums = mx.sum(axis=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        normalized_mx = mx / np.sqrt(col_sums[np.newaxis, :])

    elif mode == "row-column":
        # Row and column normalization: First row normalization, then column normalization by square root
        # Step 1: Row normalization with sqrt
        row_sums = mx.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        mx = mx / np.sqrt(row_sums[:, np.newaxis])

        # Step 2: Column normalization with sqrt
        col_sums = mx.sum(axis=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        normalized_mx = mx / np.sqrt(col_sums[np.newaxis, :])

    else:
        raise ValueError("Invalid normalization option. Choose 'row', 'column', or 'row-column'.")

    return normalized_mx