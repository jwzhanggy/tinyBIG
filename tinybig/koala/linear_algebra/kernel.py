# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
# Numerical Kernels #
#####################

from typing import Union, Any, Callable, List, Tuple
import warnings

import torch


def kernel(
    kernel_name: str,
    x: torch.Tensor = None,
    x2: torch.Tensor = None,
    *args, **kwargs
):
    """
        Selects and applies a specified kernel function.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel to apply. Supported kernels include:
            - 'linear_kernel', 'polynomial_kernel', 'hyperbolic_tangent_kernel'
            - 'exponential_kernel', 'cosine_similarity_kernel', 'euclidean_distance'
            - 'minkowski_distance', 'manhattan_distance', 'chebyshev_distance'
            - 'canberra_distance', 'gaussian_rbf_kernel', 'laplacian_kernel',
            - 'anisotropic_rbf_kernel', 'custom_hybrid_kernel'.
        x : torch.Tensor, optional
            Input tensor. Required for most kernels.
        x2 : torch.Tensor, optional
            Secondary input tensor. Required for non-batch kernels.

        Returns
        -------
        torch.Tensor
            The result of the selected kernel function.

        Raises
        ------
        ValueError
            If the input tensors are invalid.

        Examples
        --------
        >>> kernel('linear_kernel', x=torch.tensor([1.0, 2.0]), x2=torch.tensor([3.0, 4.0]))
        tensor(11.)
    """

    if 'batch' in kernel_name:
        assert x is not None and x2 is None
    else:
        assert x is not None and x2 is not None

    match kernel_name:
        case 'linear_kernel' | 'inner_product' | 'dot_product': return linear_kernel(x=x, x2=x2, *args, **kwargs)
        case 'polynomial_kernel' | 'polynomial': return polynomial_kernel(x=x, x2=x2, *args, **kwargs)
        case 'hyperbolic_tangent_kernel' | 'hyperbolic_tangent': return hyperbolic_tangent_kernel(x=x, x2=x2, *args, **kwargs)
        case 'exponential_kernel' | 'exponential': return exponential_kernel(x=x, x2=x2, *args, **kwargs)
        case 'cosine_similarity_kernel' | 'cosine_similarity': return cosine_similarity_kernel(x=x, x2=x2, *args, **kwargs)
        case 'euclidean_distance': return euclidean_distance(x=x, x2=x2, *args, **kwargs)
        case 'minkowski_distance': return minkowski_distance(x=x, x2=x2, *args, **kwargs)
        case 'manhattan_distance': return manhattan_distance(x=x, x2=x2, *args, **kwargs)
        case 'chebyshev_distance': return chebyshev_distance(x=x, x2=x2, *args, **kwargs)
        case 'canberra_distance': return canberra_distance(x=x, x2=x2, *args, **kwargs)
        case 'euclidean_distance_kernel': return euclidean_distance_kernel(x=x, x2=x2, *args, **kwargs)
        case 'minkowski_distance_kernel': return minkowski_distance_kernel(x=x, x2=x2, *args, **kwargs)
        case 'manhattan_distance_kernel': return manhattan_distance_kernel(x=x, x2=x2, *args, **kwargs)
        case 'chebyshev_distance_kernel': return chebyshev_distance_kernel(x=x, x2=x2, *args, **kwargs)
        case 'canberra_distance_kernel': return canberra_distance_kernel(x=x, x2=x2, *args, **kwargs)
        case 'gaussian_rbf_kernel' | 'gaussian_rbf': return gaussian_rbf_kernel(x=x, x2=x2, *args, **kwargs)
        case 'laplacian_kernel' | 'laplacian': return laplacian_kernel(x=x, x2=x2, *args, **kwargs)
        case 'anisotropic_rbf_kernel' | 'anisotropic_rbf' | 'anisotropic': return anisotropic_rbf_kernel(x=x, x2=x2, *args, **kwargs)
        case 'custom_hybrid_kernel' | 'custom_hybrid' | 'custom': return custom_hybrid_kernel(x=x, x2=x2, *args, **kwargs)


def linear_kernel(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    """
        Computes the linear kernel (inner product) between tensors.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        x2 : torch.Tensor, optional
            Secondary input tensor for pairwise kernel computation.
        centered : bool, default=False
            Whether to center the data before computing the kernel.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The kernel matrix or scalar value.

        Raises
        ------
        ValueError
            If input tensors are invalid.

        Examples
        --------
        >>> linear_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[ 5., 11.],
                [11., 25.]])
    """
    if x2 is None:
        return batch_linear_kernel(x, centered=centered, dim=dim)
    else:
        return instance_linear_kernel(x1=x, x2=x2)


def instance_linear_kernel(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    """
        Computes the dot product between two 1D tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.

        Returns
        -------
        torch.Tensor
            The scalar dot product.

        Raises
        ------
        ValueError
            If tensors are invalid or have mismatched dimensions.

        Examples
        --------
        >>> instance_linear_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(11.)
    """
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')
    dot_product = torch.dot(x1, x2)
    return dot_product


def batch_linear_kernel(x: torch.Tensor, centered: bool = False, dim: int = 0):
    """
        Computes the batch linear kernel matrix for a 2D tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, m).
        centered : bool, default=False
            Whether to center the data.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The linear kernel matrix of shape (m, m).

        Raises
        ------
        ValueError
            If the input tensor is invalid.

        Examples
        --------
        >>> batch_linear_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[ 5., 11.],
                [11., 25.]])
    """
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if dim == 1:
        x = x.T
    b, m = x.shape

    if centered:
        x = x - torch.mean(x, dim=0, keepdim=True)
    inner_product_matrix = torch.matmul(x.T, x)

    assert inner_product_matrix.shape == (m, m)
    return inner_product_matrix


def polynomial_kernel(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, c: float = 0.0, d: int = 1, dim: int = 0):
    """
        Computes the polynomial kernel between tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        x2 : torch.Tensor, optional
            Secondary input tensor for pairwise computation.
        centered : bool, default=False
            Whether to center the data.
        c : float, default=0.0
            Constant term in the kernel.
        d : int, default=1
            Degree of the polynomial.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The polynomial kernel matrix.

        Raises
        ------
        ValueError
            If the inputs are invalid or incompatible.

        Examples
        --------
        >>> polynomial_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), d=2, dim=1)
        tensor([[ 25., 121.],
                [121., 625.]])
    """
    if x2 is None:
        return batch_polynomial_kernel(x=x, centered=centered, c=c, d=d, dim=dim)
    else:
        return instance_polynomial_kernel(x1=x, x2=x2, c=c, d=d)


def instance_polynomial_kernel(x1: torch.Tensor, x2: torch.Tensor, c: float = 0.0, d: int = 1):
    """
        Computes the polynomial kernel for two 1D tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.
        c : float, default=0.0
            Constant term in the kernel.
        d : int, default=1
            Degree of the polynomial.

        Returns
        -------
        torch.Tensor
            The scalar polynomial kernel value.

        Raises
        ------
        ValueError
            If the inputs are invalid or incompatible.

        Examples
        --------
        >>> instance_polynomial_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), d=2)
        tensor(121.)
    """
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    dot_product = torch.dot(x1, x2)

    if dot_product + c == 0 and d < 0:
        raise ValueError("The negative powers of zeros is invalid...")

    return (dot_product + c)**d


def batch_polynomial_kernel(x: torch.Tensor, centered: bool = False, c: float = 0.0, d: int = 1, dim: int = 0):
    """
        Computes the batch polynomial kernel matrix for a 2D tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, m).
        centered : bool, default=False
            Whether to center the data.
        c : float, default=0.0
            Constant term in the kernel.
        d : int, default=1
            Degree of the polynomial.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The polynomial kernel matrix of shape (m, m).

        Raises
        ------
        ValueError
            If the input tensor is invalid.

        Examples
        --------
        >>> batch_polynomial_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), d=2, dim=1)
        tensor([[ 25., 121.],
                [121., 625.]])
    """
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if dim == 1:
        x = x.T
    b, m = x.shape

    if centered:
        x = x - torch.mean(x, dim=0, keepdim=True)
    inner_product_matrix = torch.matmul(x.T, x)

    if torch.any(inner_product_matrix + c == 0) and d < 0:
        raise ValueError("The matrix contains zero elements and the negative powers of zeros is invalid...")

    assert inner_product_matrix.shape == (m, m)
    return (inner_product_matrix + c)**d


def hyperbolic_tangent_kernel(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, c: float = 0.0, alpha: float = 1.0, dim: int = 0):
    """
        Computes the hyperbolic tangent kernel for tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        x2 : torch.Tensor, optional
            Secondary input tensor for pairwise computation.
        centered : bool, default=False
            Whether to center the data.
        c : float, default=0.0
            Bias term in the kernel.
        alpha : float, default=1.0
            Slope of the activation function.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The hyperbolic tangent kernel matrix or scalar value.

        Examples
        --------
        >>> hyperbolic_tangent_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        tensor([[1., 1.],
                [1., 1.]])
    """
    if x2 is None:
        return batch_hyperbolic_tangent_kernel(x=x, centered=centered, c=c, alpha=alpha, dim=dim)
    else:
        return instance_hyperbolic_tangent_kernel(x1=x, x2=x2, c=c, alpha=alpha)


def instance_hyperbolic_tangent_kernel(x1: torch.Tensor, x2: torch.Tensor, c: float = 0.0, alpha: float = 1.0):
    """
        Computes the hyperbolic tangent kernel for two 1D tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.
        c : float, default=0.0
            Bias term in the kernel.
        alpha : float, default=1.0
            Slope of the activation function.

        Returns
        -------
        torch.Tensor
            The scalar kernel value.

        Examples
        --------
        >>> instance_hyperbolic_tangent_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(1.)
    """
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    dot_product = torch.dot(x1, x2)

    return torch.tanh(alpha*dot_product + c)


def batch_hyperbolic_tangent_kernel(x: torch.Tensor, centered: bool = False, c: float = 0.0, alpha: float = 1.0, dim: int = 0):
    """
        Computes the batch hyperbolic tangent kernel matrix for a 2D tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, m).
        centered : bool, default=False
            Whether to center the data.
        c : float, default=0.0
            Bias term in the kernel.
        alpha : float, default=1.0
            Slope of the activation function.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The kernel matrix of shape (m, m).

        Examples
        --------
        >>> batch_hyperbolic_tangent_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        tensor([[1., 1.],
                [1., 1.]])
    """
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if dim == 1:
        x = x.T
    b, m = x.shape

    if centered:
        x = x - torch.mean(x, dim=0, keepdim=True)
    inner_product_matrix = torch.matmul(x.T, x)

    assert inner_product_matrix.shape == (m, m)
    return torch.tanh(alpha*inner_product_matrix + c)


def cosine_similarity_kernel(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    """
        Computes the cosine similarity kernel for tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        x2 : torch.Tensor, optional
            Secondary input tensor for pairwise computation.
        centered : bool, default=False
            Whether to center the data.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The cosine similarity matrix or scalar value.

        Examples
        --------
        >>> cosine_similarity_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[1.0000, 0.9839],
                [0.9839, 1.0000]])
    """
    if x2 is None:
        return batch_cosine_similarity_kernel(x=x, centered=centered, dim=dim)
    else:
        return instance_cosine_similarity_kernel(x1=x, x2=x2)


def instance_cosine_similarity_kernel(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    """
        Computes the cosine similarity between two 1D tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.

        Returns
        -------
        torch.Tensor
            The scalar similarity value.

        Examples
        --------
        >>> instance_cosine_similarity_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(0.9839)
    """
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    dot_product = torch.dot(x1, x2)
    norm_x1 = torch.norm(x1, p=2)
    norm_x2 = torch.norm(x2, p=2)

    if norm_x1 == 0 or norm_x2 == 0:
        return torch.tensor(0.0)
    else:
        cosine_sim = dot_product / (norm_x1 * norm_x2)
        return cosine_sim


def batch_cosine_similarity_kernel(x: torch.Tensor, centered: bool = False, dim: int = 0):
    """
        Computes the batch cosine similarity kernel matrix for a 2D tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, m).
        centered : bool, default=False
            Whether to center the data.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The cosine similarity matrix of shape (m, m).

        Examples
        --------
        >>> batch_cosine_similarity_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[1.0000, 0.9839],
                [0.9839, 1.0000]])
    """
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if dim == 1:
        x = x.T
    b, m = x.shape

    if centered:
        x = x - torch.mean(x, dim=0, keepdim=True)
    x_norm = torch.norm(x, p=2, dim=0, keepdim=True)
    x_norm[x_norm == 0] = torch.tensor(1.0)
    x_normalized = x / x_norm
    similarity_matrix = torch.matmul(x_normalized.T, x_normalized)

    assert similarity_matrix.shape == (m, m)
    return similarity_matrix


def minkowski_distance_kernel(x: torch.Tensor, x2: torch.Tensor = None, p: Union[int, float, str, Any] = None, centered: bool = False, dim: int = 0):
    """
        Computes the Minkowski distance kernel.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        x2 : torch.Tensor, optional
            Secondary input tensor for pairwise computation.
        p : Union[int, float, str, Any], optional
            Order of the Minkowski distance.
        centered : bool, default=False
            Whether to center the data.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The distance kernel matrix or scalar value.

        Examples
        --------
        >>> minkowski_distance_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), p=2, dim=1)
        tensor([[1.0000, 0.0591],
                [0.0591, 1.0000]])
    """
    return torch.exp(-minkowski_distance(x=x, x2=x2, p=p, centered=centered, dim=dim))


def minkowski_distance(x: torch.Tensor, x2: torch.Tensor = None, p: Union[int, float, str, Any] = None, centered: bool = False, dim: int = 0):
    """
        Computes the Minkowski distance between tensors, either as a batch or for a single pair.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        p : Union[int, float, str, Any], optional
            The order of the Minkowski distance. Must be positive or a valid norm type (e.g., 'fro', 'nuc').
        centered : bool, default=False
            Whether to center the data when computing distances in batch mode.
        dim : int, default=0
            The dimension along which the distances are computed when in batch mode.

        Returns
        -------
        torch.Tensor
            - For pairwise computation (x and x2 provided): A scalar value representing the Minkowski distance.
            - For batch computation (x2 is None): A distance matrix of shape (m, m).

        Examples
        --------
        Pairwise Minkowski distance:
        >>> minkowski_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), p=2)
        tensor(2.8284)

        Batch Minkowski distance:
        >>> minkowski_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), p=2, dim=1)
        tensor([[0.0000, 2.8284],
                [2.8284, 0.0000]])
    """
    if x2 is None:
        return batch_minkowski_distance(x=x, p=p, centered=centered, dim=dim)
    else:
        return instance_minkowski_distance(x1=x, x2=x2, p=p)


def instance_minkowski_distance(x1: torch.Tensor, x2: torch.Tensor, p: Union[int, float, str, Any] = None):
    """
        Computes the Minkowski distance between two 1D tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.
        p : Union[int, float, str, Any], optional
            The order of the Minkowski distance. Must be positive or a valid norm type.

        Returns
        -------
        torch.Tensor
            The Minkowski distance as a scalar value.

        Examples
        --------
        >>> instance_minkowski_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), p=2)
        tensor(2.8284)
    """
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')
    if p is None:
        raise ValueError('p must be provided and cannot be None...')
    if isinstance(p, str) and p not in ['fro', 'nuc']:
        raise ValueError('p must be either "fro" or "nuc" or torch.inf (numpy.inf) or -torch.inf (numpy.inf) or numbers...')

    distance = torch.norm(x1 - x2, p=p)

    return distance


def batch_minkowski_distance(x: torch.Tensor, p: Union[int, float, str, Any] = None, centered: bool = False, dim: int = 0):
    """
        Computes the Minkowski distance matrix for a batch of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, m).
        p : Union[int, float, str, Any], optional
            The order of the Minkowski distance. Must be positive or a valid norm type.
        centered : bool, default=False
            Whether to center the data.
        dim : int, default=0
            The dimension along which the distances are computed.

        Returns
        -------
        torch.Tensor
            A distance matrix of shape (m, m).

        Examples
        --------
        >>> batch_minkowski_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), p=2)
        tensor([[0.0000, 1.4142],
                [1.4142, 0.0000]])

        >>> batch_minkowski_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), p=2, dim=1)
        tensor([[0.0000, 2.8284],
                [2.8284, 0.0000]])
    """
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')
    if p is None:
        raise ValueError('p must be provided and cannot be None...')
    if isinstance(p, str) and p not in ['fro', 'nuc']:
        raise ValueError('p must be either "fro" or "nuc" or torch.inf (numpy.inf) or -torch.inf (numpy.inf) or numbers...')

    if dim == 1:
        x = x.T
    b, m = x.shape

    if centered:
        x = x - torch.mean(x, dim=0, keepdim=True)
    x_expanded_1 = x.unsqueeze(2)
    x_expanded_2 = x.unsqueeze(1)
    distance_matrix = torch.norm(x_expanded_1 - x_expanded_2, p=p, dim=0)

    assert distance_matrix.shape == (m, m)
    return distance_matrix


def manhattan_distance_kernel(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    """
        Computes the Manhattan distance kernel, which is the exponential of the negative Manhattan distance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        centered : bool, default=False
            Whether to center the data when computing distances in batch mode.
        dim : int, default=0
            The dimension along which the distances are computed when in batch mode.

        Returns
        -------
        torch.Tensor
            The Manhattan distance kernel. Shape depends on whether x2 is provided or not:
            - If x2 is provided: A scalar kernel value.
            - If x2 is None: A kernel matrix of shape (m, m).

        Examples
        --------
        >>> manhattan_distance_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(0.0183)

        >>> manhattan_distance_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[1.0000, 0.0183],
                [0.0183, 1.0000]])
    """
    return torch.exp(-manhattan_distance(x=x, x2=x2, centered=centered, dim=dim))


def manhattan_distance(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    """
        Computes the Manhattan distance between tensors, either as a batch or for a single pair.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        centered : bool, default=False
            Whether to center the data when computing distances in batch mode.
        dim : int, default=0
            The dimension along which the distances are computed when in batch mode.

        Returns
        -------
        torch.Tensor
            - For pairwise computation (x and x2 provided): A scalar distance value.
            - For batch computation (x2 is None): A distance matrix of shape (m, m).

        Examples
        --------
        Pairwise Manhattan distance:
        >>> manhattan_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(4.)

        Batch Manhattan distance:
        >>> manhattan_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[0., 4.],
                [4., 0.]])
    """
    if x2 is None:
        return batch_manhattan_distance(x=x, centered=centered, dim=dim)
    else:
        return instance_manhattan_distance(x1=x, x2=x2)


def instance_manhattan_distance(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    """
        Computes the Manhattan distance (L1 norm) between two tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor. Must be 1-dimensional.
        x2 : torch.Tensor
            The second input tensor. Must be 1-dimensional and have the same shape as x1.

        Returns
        -------
        torch.Tensor
            The Manhattan distance between x1 and x2.

        Examples
        --------
        >>> instance_manhattan_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(4.)
    """
    return instance_minkowski_distance(x1=x1, x2=x2, p=1)


def batch_manhattan_distance(x: torch.Tensor, centered: bool = False, dim: int = 0):
    """
        Computes the Manhattan distance (L1 norm) between all pairs of rows in a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Must be 2-dimensional.
        centered : bool, default=False
            Whether to center the data when computing distances.
        dim : int, default=0
            The dimension along which the distances are computed.

        Returns
        -------
        torch.Tensor
            A matrix of pairwise Manhattan distances with shape (m, m), where m is the number of rows in x.

        Examples
        --------
        >>> batch_manhattan_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[0., 4.],
                [4., 0.]])
    """
    return batch_minkowski_distance(x=x, p=1, centered=centered, dim=dim)


def euclidean_distance_kernel(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    """
        Computes the Euclidean distance kernel, which is the exponential of the negative Euclidean distance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        centered : bool, default=False
            Whether to center the data when computing distances in batch mode.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            The Euclidean distance kernel. Shape depends on whether x2 is provided:
            - If x2 is provided: A scalar kernel value.
            - If x2 is None: A kernel matrix of shape (m, m).

        Examples
        --------
        >>> euclidean_distance_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(0.0591)

        >>> euclidean_distance_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[1.0000, 0.0591],
                [0.0591, 1.0000]])
    """
    return torch.exp(-euclidean_distance(x=x, x2=x2, centered=centered, dim=dim))


def euclidean_distance(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    """
        Computes the Euclidean distance between tensors, either as a batch or for a single pair.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        centered : bool, default=False
            Whether to center the data when computing distances in batch mode.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            - For pairwise computation (x and x2 provided): A scalar distance value.
            - For batch computation (x2 is None): A distance matrix of shape (m, m).

        Examples
        --------
        Pairwise Euclidean distance:
        >>> euclidean_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(2.8284)

        Batch Euclidean distance:
        >>> euclidean_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[0.0000, 2.8284],
                [2.8284, 0.0000]])
    """
    if x2 is None:
        return batch_euclidean_distance(x=x, centered=centered, dim=dim)
    else:
        return instance_euclidean_distance(x1=x, x2=x2)


def instance_euclidean_distance(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    """
        Computes the Euclidean distance (L2 norm) between two tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor. Must be 1-dimensional.
        x2 : torch.Tensor
            The second input tensor. Must be 1-dimensional and have the same shape as x1.

        Returns
        -------
        torch.Tensor
            The Euclidean distance between x1 and x2.

        Examples
        --------
        >>> instance_euclidean_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(2.8284)
    """
    return instance_minkowski_distance(x1=x1, x2=x2, p=2)


def batch_euclidean_distance(x: torch.Tensor, centered: bool = False, dim: int = 0):
    """
        Computes the Euclidean distance (L2 norm) between all pairs of rows in a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Must be 2-dimensional.
        centered : bool, default=False
            Whether to center the data when computing distances.
        dim : int, default=0
            The dimension along which the distances are computed.

        Returns
        -------
        torch.Tensor
            A matrix of pairwise Euclidean distances with shape (m, m), where m is the number of rows in x.

        Examples
        --------
        >>> batch_euclidean_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[0.0000, 2.8284],
                [2.8284, 0.0000]])
    """
    return batch_minkowski_distance(x=x, p=2, centered=centered, dim=dim)


def chebyshev_distance_kernel(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    """
        Computes the Chebyshev distance kernel, which is the exponential of the negative Chebyshev distance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        centered : bool, default=False
            Whether to center the data when computing distances in batch mode.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            The Chebyshev distance kernel. Shape depends on whether x2 is provided:
            - If x2 is provided: A scalar kernel value.
            - If x2 is None: A kernel matrix of shape (m, m).

        Examples
        --------
        Pairwise Chebyshev distance kernel:
        >>> chebyshev_distance_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(0.1353)

        Batch Chebyshev distance kernel:
        >>> chebyshev_distance_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[1.0000, 0.1353],
                [0.1353, 1.0000]])
    """
    return torch.exp(-chebyshev_distance(x=x, x2=x2, centered=centered, dim=dim))


def chebyshev_distance(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    """
        Computes the Chebyshev distance between tensors, either as a batch or for a single pair.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        centered : bool, default=False
            Whether to center the data when computing distances in batch mode.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            - For pairwise computation (x and x2 provided): A scalar distance value.
            - For batch computation (x2 is None): A distance matrix of shape (m, m).

        Examples
        --------
        Pairwise Chebyshev distance:
        >>> chebyshev_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(2.)

        Batch Chebyshev distance:
        >>> chebyshev_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[0., 2.],
                [2., 0.]])
    """
    if x2 is None:
        return batch_chebyshev_distance(x=x, centered=centered, dim=dim)
    else:
        return instance_chebyshev_distance(x1=x, x2=x2)


def instance_chebyshev_distance(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    """
        Computes the Chebyshev distance between two 1D tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.

        Returns
        -------
        torch.Tensor
            The Chebyshev distance as a scalar value.

        Examples
        --------
        >>> instance_chebyshev_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 5.0]))
        tensor(3.)
    """
    return instance_minkowski_distance(x1=x1, x2=x2, p=torch.inf)


def batch_chebyshev_distance(x: torch.Tensor, centered: bool = False, dim: int = 0):
    """
        Computes the Chebyshev distance matrix for a batch of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, m).
        centered : bool, default=False
            Whether to center the data.
        dim : int, default=0
            The dimension along which the distances are computed.

        Returns
        -------
        torch.Tensor
            A distance matrix of shape (m, m).

        Examples
        --------
        >>> batch_chebyshev_distance(torch.tensor([[1.0, 2.0], [3.0, 5.0]]), dim=1)
        tensor([[0., 3.],
                [3., 0.]])
    """
    return batch_minkowski_distance(x=x, p=torch.inf, centered=centered, dim=dim)


def canberra_distance_kernel(x: torch.Tensor, x2: torch.Tensor = None, dim: int = 0):
    """
        Computes the Canberra distance kernel, which is the exponential of the negative Canberra distance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            The Canberra distance kernel. Shape depends on whether x2 is provided:
            - If x2 is provided: A scalar kernel value.
            - If x2 is None: A kernel matrix of shape (m, m).

        Examples
        --------
        Pairwise Canberra distance kernel:
        >>> canberra_distance_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(0.4346)

        Batch Canberra distance kernel:
        >>> canberra_distance_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[1.0000, 0.4346],
                [0.4346, 1.0000]])
    """
    return torch.exp(-canberra_distance(x=x, x2=x2, dim=dim))


def canberra_distance(x: torch.Tensor, x2: torch.Tensor = None, dim: int = 0):
    """
        Computes the Canberra distance between tensors, either as a batch or for a single pair.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            - For pairwise computation (x and x2 provided): A scalar distance value.
            - For batch computation (x2 is None): A distance matrix of shape (m, m).

        Examples
        --------
        Pairwise Canberra distance:
        >>> canberra_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(0.8333)

        Batch Canberra distance:
        >>> canberra_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[0.0000, 0.8333],
                [0.8333, 0.0000]])
    """
    if x2 is None:
        return batch_canberra_distance(x=x, dim=dim)
    else:
        return instance_canberra_distance(x1=x, x2=x2)


def instance_canberra_distance(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    """
        Computes the Canberra distance between two 1D tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.

        Returns
        -------
        torch.Tensor
            The Canberra distance as a scalar value.

        Examples
        --------
        >>> instance_canberra_distance(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(0.8333)
    """
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    numerator = torch.absolute(x1 - x2)
    denominator = torch.absolute(x1) + torch.absolute(x2)
    canberra_dist = torch.sum(numerator / (denominator + 1e-10))

    return canberra_dist


def batch_canberra_distance(x: torch.Tensor, dim: int = 0):
    """
        Computes the Canberra distance matrix for a batch of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, m).
        dim : int, default=0
            The dimension along which the distances are computed.

        Returns
        -------
        torch.Tensor
            A distance matrix of shape (m, m).

        Examples
        --------
        >>> batch_canberra_distance(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
        tensor([[0.0000, 0.8333],
                [0.8333, 0.0000]])
    """
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if dim == 1:
        x = x.T
    b, m = x.shape

    x_expanded_1 = torch.unsqueeze(x, dim=2)
    x_expanded_2 = torch.unsqueeze(x, dim=1)

    numerator = torch.absolute(x_expanded_1 - x_expanded_2)
    denominator = torch.absolute(x_expanded_1) + torch.absolute(x_expanded_2)
    canberra_dist_matrix = torch.sum(numerator / (denominator + 1e-10), dim=0)

    assert canberra_dist_matrix.shape == (m, m)
    return canberra_dist_matrix


def exponential_kernel(x: torch.Tensor, x2: torch.Tensor = None, gamma: float = 1.0, dim: int = 0):
    """
        Computes the exponential kernel, which is an RBF-like kernel based on the exponential of squared Euclidean distance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, distances are computed in batch mode.
        gamma : float, default=1.0
            The scale parameter for the kernel.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            - For pairwise computation (x and x2 provided): A scalar kernel value.
            - For batch computation (x2 is None): A kernel matrix of shape (m, m).

        Examples
        --------
        Pairwise exponential kernel:
        >>> exponential_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), gamma=0.5)
        tensor(0.0183)

        Batch exponential kernel:
        >>> exponential_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), gamma=0.5, dim=1)
        tensor([[1.0000, 0.0183],
                [0.0183, 1.0000]])
    """
    if x2 is None:
        return batch_exponential_kernel(x=x, gamma=gamma, dim=dim)
    else:
        return instance_exponential_kernel(x1=x, x2=x2, gamma=gamma)


def instance_exponential_kernel(x1: torch.Tensor, x2: torch.Tensor, gamma: float = 1.0):
    """
        Computes the exponential kernel for a single pair of tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.
        gamma : float, default=1.0
            The scale parameter for the kernel.

        Returns
        -------
        torch.Tensor
            The kernel value as a scalar tensor.

        Examples
        --------
        >>> instance_exponential_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), gamma=0.5)
        tensor(0.0183)
    """
    return torch.exp(-gamma*instance_euclidean_distance(x1=x1, x2=x2)**2)


def batch_exponential_kernel(x: torch.Tensor, gamma: float = 1.0, dim: int = 0):
    """
        Computes the exponential kernel in batch mode for a set of tensors.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Must be a 2D tensor with shape (b, m).
        gamma : float, default=1.0
            The scale parameter for the kernel.
        dim : int, default=0
            The dimension along which the distances are computed.

        Returns
        -------
        torch.Tensor
            A kernel matrix of shape (m, m).

        Examples
        --------
        >>> batch_exponential_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), gamma=0.5, dim=1)
        tensor([[1.0000, 0.0183],
                [0.0183, 1.0000]])
    """
    return torch.exp(-gamma*batch_euclidean_distance(x=x, dim=dim)**2)


def gaussian_rbf_kernel(x: torch.Tensor, x2: torch.Tensor = None, sigma: float = 1.0, dim: int = 0):
    """
        Computes the Gaussian Radial Basis Function (RBF) kernel.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, the kernel is computed in batch mode.
        sigma : float, default=1.0
            The standard deviation parameter for the RBF kernel.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            - For pairwise computation (x and x2 provided): A scalar kernel value.
            - For batch computation (x2 is None): A kernel matrix of shape (m, m).

        Raises
        ------
        ValueError
            If `sigma` is less than or equal to zero.

        Examples
        --------
        Pairwise Gaussian RBF kernel:
        >>> gaussian_rbf_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), sigma=1.0)
        tensor(0.0183)

        Batch Gaussian RBF kernel:
        >>> gaussian_rbf_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), sigma=1.0, dim=1)
        tensor([[1.0000, 0.0183],
                [0.0183, 1.0000]])
    """
    if x2 is None:
        return batch_gaussian_rbf_kernel(x=x, sigma=sigma, dim=dim)
    else:
        return instance_gaussian_rbf_kernel(x1=x, x2=x2, sigma=sigma)


def instance_gaussian_rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, sigma: float = 1.0):
    """
        Computes the Gaussian Radial Basis Function (RBF) kernel for two 1D tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.
        sigma : float, default=1.0
            The standard deviation of the Gaussian kernel.

        Returns
        -------
        torch.Tensor
            The Gaussian RBF kernel value.

        Examples
        --------
        >>> instance_gaussian_rbf_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), sigma=1.0)
        tensor(0.0183)
    """
    if sigma <= 0.0:
        raise ValueError('sigma must be positive...')
    return torch.exp(- instance_euclidean_distance(x1=x1, x2=x2)**2 / (2 * sigma**2))


def batch_gaussian_rbf_kernel(x: torch.Tensor, sigma: float = 1.0, dim: int = 0):
    """
        Computes the Gaussian Radial Basis Function (RBF) kernel matrix for a batch of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, m).
        sigma : float, default=1.0
            The standard deviation of the Gaussian kernel.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The Gaussian RBF kernel matrix of shape (m, m).

        Examples
        --------
        >>> batch_gaussian_rbf_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), sigma=1.0, dim=1)
        tensor([[1.0000, 0.0183],
                [0.0183, 1.0000]])
    """
    if sigma <= 0.0:
        raise ValueError('sigma must be positive...')
    return torch.exp(- batch_euclidean_distance(x=x, dim=dim) ** 2 / (2 * sigma ** 2))


def laplacian_kernel(x: torch.Tensor, x2: torch.Tensor = None, sigma: float = 1.0, dim: int = 0):
    """
        Computes the Laplacian kernel.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, the kernel is computed in batch mode.
        sigma : float, default=1.0
            The scale parameter for the Laplacian kernel.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            - For pairwise computation (x and x2 provided): A scalar kernel value.
            - For batch computation (x2 is None): A kernel matrix of shape (m, m).

        Raises
        ------
        ValueError
            If `sigma` is zero.

        Examples
        --------
        Pairwise Laplacian kernel:
        >>> laplacian_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), sigma=1.0)
        tensor(0.0183)

        Batch Laplacian kernel:
        >>> laplacian_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), sigma=1.0, dim=1)
        tensor([[1.0000, 0.0183],
                [0.0183, 1.0000]])
    """
    if x2 is None:
        return batch_laplacian_kernel(x=x, sigma=sigma, dim=dim)
    else:
        return instance_laplacian_kernel(x1=x, x2=x2, sigma=sigma)


def instance_laplacian_kernel(x1: torch.Tensor, x2: torch.Tensor, sigma: float = 1.0):
    """
        Computes the Laplacian kernel for two 1D tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor.
        x2 : torch.Tensor
            The second input tensor.
        sigma : float, default=1.0
            The kernel bandwidth parameter.

        Returns
        -------
        torch.Tensor
            The Laplacian kernel value.

        Examples
        --------
        >>> instance_laplacian_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), sigma=1.0)
        tensor(0.0183)
    """
    if sigma == 0:
        raise ValueError('sigma must be not be zero...')
    return torch.exp(- instance_manhattan_distance(x1=x1, x2=x2)/sigma)


def batch_laplacian_kernel(x: torch.Tensor, sigma: float = 1.0, dim: int = 0):
    """
        Computes the Laplacian kernel matrix for a batch of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (b, m).
        sigma : float, default=1.0
            The kernel bandwidth parameter.
        dim : int, default=0
            The dimension along which the kernel is computed.

        Returns
        -------
        torch.Tensor
            The Laplacian kernel matrix of shape (m, m).

        Examples
        --------
        >>> batch_laplacian_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), sigma=1.0, dim=1)
        tensor([[1.0000, 0.0183],
                [0.0183, 1.0000]])
    """
    if sigma == 0:
        raise ValueError('sigma must be not be zero...')
    return torch.exp(- batch_manhattan_distance(x=x, dim=dim)/sigma)


def anisotropic_rbf_kernel(x: torch.Tensor, x2: torch.Tensor = None, a_vector: torch.Tensor = None, a_scalar: float = 1.0, dim: int = 0):
    """
        Computes the anisotropic radial basis function (RBF) kernel.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. For pairwise computation, this is the first tensor (x1).
        x2 : torch.Tensor, optional
            The second input tensor for pairwise computation. If not provided, the kernel is computed in batch mode.
        a_vector : torch.Tensor, optional
            A vector defining the anisotropy in each dimension. Must have the same shape as `x`.
        a_scalar : float, default=1.0
            A scalar defining the uniform anisotropy. Ignored if `a_vector` is provided.
        dim : int, default=0
            The dimension along which the distances are computed in batch mode.

        Returns
        -------
        torch.Tensor
            - For pairwise computation (x and x2 provided): A scalar kernel value.
            - For batch computation (x2 is None): A kernel matrix of shape (m, m).

        Raises
        ------
        ValueError
            If input tensors are empty or `a_vector` is incompatible with `x`.
        Warnings
            If `a_vector` is all zeros or `a_scalar` is zero.

        Examples
        --------
        Pairwise anisotropic RBF kernel:
        >>> anisotropic_rbf_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), a_vector=torch.tensor([0.5, 0.5]))
        tensor(0.0183)

        Batch anisotropic RBF kernel:
        >>> anisotropic_rbf_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), a_scalar=0.5, dim=1)
        tensor([[1.0000, 0.0183],
                [0.0183, 1.0000]])
    """
    if x2 is None:
        return batch_anisotropic_rbf_kernel(x=x, a_vector=a_vector, a_scalar=a_scalar, dim=dim)
    else:
        return instance_anisotropic_rbf_kernel(x1=x, x2=x2, a_vector=a_vector, a_scalar=a_scalar)


def instance_anisotropic_rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, a_vector: torch.Tensor = None, a_scalar: float = 1.0):
    """
        Computes the pairwise anisotropic RBF kernel for two input tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor (1D).
        x2 : torch.Tensor
            The second input tensor (1D).
        a_vector : torch.Tensor, optional
            A vector defining the anisotropy in each dimension. Must have the same shape as `x1`.
        a_scalar : float, default=1.0
            A scalar defining the uniform anisotropy. Ignored if `a_vector` is provided.

        Returns
        -------
        torch.Tensor
            A scalar value of the kernel.

        Raises
        ------
        ValueError
            If input tensors are empty or dimensions do not match.
        Warnings
            If `a_vector` is all zeros or `a_scalar` is zero.

        Examples
        --------
        >>> instance_anisotropic_rbf_kernel(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), a_vector=torch.tensor([0.5, 0.5]))
        tensor(0.0183)
    """
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')
    if a_vector is not None and a_vector.shape != x1.shape:
        raise ValueError('a_vector must be of dimension 1 and has identical shapes as x1 and x2...')
    if a_vector is not None and a_vector.numel() == 0:
        raise ValueError("Input a vector must not be None or empty...")

    if a_vector is not None and torch.all(a_vector == 0):
        warnings.warn('input a vector should not be all zeros...')
    if a_scalar is not None and a_scalar == 0.0:
        warnings.warn("Input a scalar must not be zero...")

    if a_vector is not None:
        a = a_vector
    else:
        a = a_scalar * torch.ones_like(x1)

    assert a.shape == x1.shape

    A = torch.diag(a)
    d = x1 - x2
    return torch.exp(-d @ A @ d.permute(*torch.arange(d.ndim - 1, -1, -1)))


def batch_anisotropic_rbf_kernel(x: torch.Tensor, a_vector: torch.Tensor = None, a_scalar: float = 1.0, dim: int = 0):
    """
        Computes the batch anisotropic RBF kernel.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (2D).
        a_vector : torch.Tensor, optional
            A vector defining the anisotropy in each dimension. Must match the batch size of `x`.
        a_scalar : float, default=1.0
            A scalar defining the uniform anisotropy. Ignored if `a_vector` is provided.
        dim : int, default=0
            The dimension along which the distances are computed.

        Returns
        -------
        torch.Tensor
            A kernel matrix of shape (m, m).

        Raises
        ------
        ValueError
            If input tensors are empty or incompatible.
        Warnings
            If `a_vector` is all zeros or `a_scalar` is zero.

        Examples
        --------
        >>> batch_anisotropic_rbf_kernel(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), a_scalar=0.5, dim=1)
        tensor([[1.0000, 0.0183],
                [0.0183, 1.0000]])
    """
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')
    if a_vector is not None and a_vector.numel() == 0:
        raise ValueError("Input a vector must not be None or empty...")

    if a_vector is not None and torch.all(a_vector == 0):
        warnings.warn('input a vector should not be all zeros...')
    if a_scalar is not None and a_scalar == 0.0:
        warnings.warn("Input a scalar must not be zero...")

    if dim == 1:
        x = x.T
    b, m = x.shape

    if a_vector is not None:
        a = a_vector
    else:
        a = a_scalar * torch.ones(b)

    assert a is not None and a.ndim == 1 and a.size(0) == b

    A = torch.diag(a)

    x = x.T  # Shape: (m, b)

    diff_matrix = x[:, None, :] - x[None, :, :]  # Shape: (m, m, b)
    rbf_matrix = torch.einsum('ijk,kl,ijl->ij', diff_matrix, A, diff_matrix)

    assert rbf_matrix.shape == (m, m)
    return torch.exp(-rbf_matrix)


def custom_hybrid_kernel(x: torch.Tensor, x2: torch.Tensor = None, kernels: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, weights: Union[List, Tuple, float, torch.nn.Parameter] = None, dim: int = 0):
    """
        Computes a custom hybrid kernel by combining multiple kernel functions with specified weights.

        Parameters
        ----------
        x : torch.Tensor
            The first input tensor. Can be a batch or single instance depending on `x2`.
        x2 : torch.Tensor, optional
            The second input tensor. If `None`, the kernel will be computed in batch mode.
        kernels : List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            A list of kernel functions to be combined.
        weights : Union[List, Tuple, float, torch.nn.Parameter], optional
            Weights for combining the kernel functions. If `None`, all kernels are equally weighted.
        dim : int, optional
            Dimension to apply the kernel operation. Default is `0`.

        Returns
        -------
        torch.Tensor
            The computed hybrid kernel output.

        Raises
        ------
        ValueError
            If input tensors are invalid, no kernels are provided, or weights and kernels mismatch.
    """
    if x2 is None:
        return batch_custom_hybrid_kernel(x=x, kernels=kernels, weights=weights, dim=dim)
    else:
        return instance_custom_hybrid_kernel(x1=x, x2=x2, kernels=kernels, weights=weights)


def instance_custom_hybrid_kernel(x1: torch.Tensor, x2: torch.Tensor, kernels: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], weights: Union[List, Tuple, float] = None):
    """
        Computes a hybrid kernel for two single instances using a combination of kernel functions.

        Parameters
        ----------
        x1 : torch.Tensor
            The first input tensor (single instance).
        x2 : torch.Tensor
            The second input tensor (single instance).
        kernels : List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            A list of kernel functions to be combined.
        weights : Union[List, Tuple, float], optional
            Weights for combining the kernel functions. If `None`, all kernels are equally weighted.

        Returns
        -------
        torch.Tensor
            The computed hybrid kernel value for the two instances.

        Raises
        ------
        ValueError
            If input tensors are invalid, no kernels are provided, or weights and kernels mismatch.
    """
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    if kernels is None or len(kernels) == 0:
        raise ValueError("At least one kernel function must be provided.")
    elif not isinstance(kernels, list):
        kernels = [kernels]

    if weights is None:
        weights = [1 / len(kernels)] * len(kernels)
    elif not isinstance(weights, list):
        weights = [weights]

    if len(kernels) != len(weights):
        raise ValueError("The number of kernels must match the number of weights.")

    kernel_outputs = [kernel(x1, x2) for kernel in kernels]
    shapes = [output.shape for output in kernel_outputs]
    if len(set(shapes)) != 1:
        raise ValueError("All kernel outputs must have the same shape.")

    weighted_sum = sum(weight * output for weight, output in zip(weights, kernel_outputs))
    return weighted_sum


def batch_custom_hybrid_kernel(x: torch.Tensor, kernels: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], weights: Union[List, Tuple, float] = None, dim: int = 0):
    """
        Computes a hybrid kernel for a batch of instances using a combination of kernel functions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, feature_dim)` for batch processing.
        kernels : List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            A list of kernel functions to be combined.
        weights : Union[List, Tuple, float], optional
            Weights for combining the kernel functions. If `None`, all kernels are equally weighted.
        dim : int, optional
            Dimension to apply the kernel operation. Default is `0`.

        Returns
        -------
        torch.Tensor
            The computed hybrid kernel matrix for the batch.

        Raises
        ------
        ValueError
            If input tensors are invalid, no kernels are provided, or weights and kernels mismatch.
    """
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if kernels is None or len(kernels) == 0:
        raise ValueError("At least one kernel function must be provided.")
    elif not isinstance(kernels, list):
        kernels = [kernels]

    if weights is None:
        weights = [1 / len(kernels)] * len(kernels)
    elif not isinstance(weights, list):
        weights = [weights]

    if len(kernels) != len(weights):
        raise ValueError("The number of kernels must match the number of weights.")

    kernel_outputs = [kernel(x, dim=dim) for kernel in kernels]

    shapes = [output.shape for output in kernel_outputs]
    if len(set(shapes)) != 1:
        raise ValueError("All kernel outputs must have the same shape.")

    weighted_sum = sum(weight * output for weight, output in zip(weights, kernel_outputs))
    return weighted_sum

