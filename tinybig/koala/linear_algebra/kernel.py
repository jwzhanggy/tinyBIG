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
    if x2 is None:
        return batch_linear_kernel(x, centered=centered, dim=dim)
    else:
        return instance_linear_kernel(x1=x, x2=x2)


def instance_linear_kernel(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')
    dot_product = torch.dot(x1, x2)
    return dot_product


def batch_linear_kernel(x: torch.Tensor, centered: bool = False, dim: int = 0):
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
    if x2 is None:
        return batch_polynomial_kernel(x=x, centered=centered, c=c, d=d, dim=dim)
    else:
        return instance_polynomial_kernel(x1=x, x2=x2, c=c, d=d)


def instance_polynomial_kernel(x1: torch.Tensor, x2: torch.Tensor, c: float = 0.0, d: int = 1):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    dot_product = torch.dot(x1, x2)

    if dot_product + c == 0 and d < 0:
        raise ValueError("The negative powers of zeros is invalid...")

    return (dot_product + c)**d


def batch_polynomial_kernel(x: torch.Tensor, centered: bool = False, c: float = 0.0, d: int = 1, dim: int = 0):
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
    if x2 is None:
        return batch_hyperbolic_tangent_kernel(x=x, centered=centered, c=c, alpha=alpha, dim=dim)
    else:
        return instance_hyperbolic_tangent_kernel(x1=x, x2=x2, c=c, alpha=alpha)


def instance_hyperbolic_tangent_kernel(x1: torch.Tensor, x2: torch.Tensor, c: float = 0.0, alpha: float = 1.0):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    dot_product = torch.dot(x1, x2)

    return torch.tanh(alpha*dot_product + c)


def batch_hyperbolic_tangent_kernel(x: torch.Tensor, centered: bool = False, c: float = 0.0, alpha: float = 1.0, dim: int = 0):
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
    if x2 is None:
        return batch_cosine_similarity_kernel(x=x, centered=centered, dim=dim)
    else:
        return instance_cosine_similarity_kernel(x1=x, x2=x2)


def instance_cosine_similarity_kernel(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
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
    return torch.exp(-minkowski_distance(x=x, x2=x2, p=p, centered=centered, dim=dim))


def minkowski_distance(x: torch.Tensor, x2: torch.Tensor = None, p: Union[int, float, str, Any] = None, centered: bool = False, dim: int = 0):
    if x2 is None:
        return batch_minkowski_distance(x=x, p=p, centered=centered, dim=dim)
    else:
        return instance_minkowski_distance(x1=x, x2=x2, p=p)


def instance_minkowski_distance(x1: torch.Tensor, x2: torch.Tensor, p: Union[int, float, str, Any] = None):
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
    return torch.exp(-manhattan_distance(x=x, x2=x2, centered=centered, dim=dim))


def manhattan_distance(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    if x2 is None:
        return batch_manhattan_distance(x=x, centered=centered, dim=dim)
    else:
        return instance_manhattan_distance(x1=x, x2=x2)


def instance_manhattan_distance(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    return instance_minkowski_distance(x1=x1, x2=x2, p=1)


def batch_manhattan_distance(x: torch.Tensor, centered: bool = False, dim: int = 0):
    return batch_minkowski_distance(x=x, p=1, centered=centered, dim=dim)


def euclidean_distance_kernel(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    return torch.exp(-euclidean_distance(x=x, x2=x2, centered=centered, dim=dim))


def euclidean_distance(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    if x2 is None:
        return batch_euclidean_distance(x=x, centered=centered, dim=dim)
    else:
        return instance_euclidean_distance(x1=x, x2=x2)


def instance_euclidean_distance(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    return instance_minkowski_distance(x1=x1, x2=x2, p=2)


def batch_euclidean_distance(x: torch.Tensor, centered: bool = False, dim: int = 0):
    return batch_minkowski_distance(x=x, p=2, centered=centered, dim=dim)


def chebyshev_distance_kernel(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    return torch.exp(-chebyshev_distance(x=x, x2=x2, centered=centered, dim=dim))


def chebyshev_distance(x: torch.Tensor, x2: torch.Tensor = None, centered: bool = False, dim: int = 0):
    if x2 is None:
        return batch_chebyshev_distance(x=x, centered=centered, dim=dim)
    else:
        return instance_chebyshev_distance(x1=x, x2=x2)


def instance_chebyshev_distance(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    return instance_minkowski_distance(x1=x1, x2=x2, p=torch.inf)


def batch_chebyshev_distance(x: torch.Tensor, centered: bool = False, dim: int = 0):
    return batch_minkowski_distance(x=x, p=torch.inf, centered=centered, dim=dim)


def canberra_distance_kernel(x: torch.Tensor, x2: torch.Tensor = None, dim: int = 0):
    return torch.exp(-canberra_distance(x=x, x2=x2, dim=dim))


def canberra_distance(x: torch.Tensor, x2: torch.Tensor = None, dim: int = 0):
    if x2 is None:
        return batch_canberra_distance(x=x, dim=dim)
    else:
        return instance_canberra_distance(x1=x, x2=x2)


def instance_canberra_distance(x1: torch.Tensor, x2: torch.Tensor, *args, **kwargs):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty...")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    numerator = torch.absolute(x1 - x2)
    denominator = torch.absolute(x1) + torch.absolute(x2)
    canberra_dist = torch.sum(numerator / (denominator + 1e-10))

    return canberra_dist


def batch_canberra_distance(x: torch.Tensor, dim: int = 0):
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
    if x2 is None:
        return batch_exponential_kernel(x=x, gamma=gamma, dim=dim)
    else:
        return instance_exponential_kernel(x1=x, x2=x2, gamma=gamma)


def instance_exponential_kernel(x1: torch.Tensor, x2: torch.Tensor, gamma: float = 1.0):
    return torch.exp(-gamma*instance_euclidean_distance(x1=x1, x2=x2)**2)


def batch_exponential_kernel(x: torch.Tensor, gamma: float = 1.0, dim: int = 0):
    return torch.exp(-gamma*batch_euclidean_distance(x=x, dim=dim)**2)


def gaussian_rbf_kernel(x: torch.Tensor, x2: torch.Tensor = None, sigma: float = 1.0, dim: int = 0):
    if x2 is None:
        return batch_gaussian_rbf_kernel(x=x, sigma=sigma, dim=dim)
    else:
        return instance_gaussian_rbf_kernel(x1=x, x2=x2, sigma=sigma)


def instance_gaussian_rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, sigma: float = 1.0):
    if sigma <= 0.0:
        raise ValueError('sigma must be positive...')
    return torch.exp(- instance_euclidean_distance(x1=x1, x2=x2)**2 / (2 * sigma**2))


def batch_gaussian_rbf_kernel(x: torch.Tensor, sigma: float = 1.0, dim: int = 0):
    if sigma <= 0.0:
        raise ValueError('sigma must be positive...')
    return torch.exp(- batch_euclidean_distance(x=x, dim=dim) ** 2 / (2 * sigma ** 2))


def laplacian_kernel(x: torch.Tensor, x2: torch.Tensor = None, sigma: float = 1.0, dim: int = 0):
    if x2 is None:
        return batch_laplacian_kernel(x=x, sigma=sigma, dim=dim)
    else:
        return instance_laplacian_kernel(x1=x, x2=x2, sigma=sigma)


def instance_laplacian_kernel(x1: torch.Tensor, x2: torch.Tensor, sigma: float = 1.0):
    if sigma == 0:
        raise ValueError('sigma must be not be zero...')
    return torch.exp(- instance_manhattan_distance(x1=x1, x2=x2)/sigma)


def batch_laplacian_kernel(x: torch.Tensor, sigma: float = 1.0, dim: int = 0):
    if sigma == 0:
        raise ValueError('sigma must be not be zero...')
    return torch.exp(- batch_manhattan_distance(x=x, dim=dim)/sigma)


def anisotropic_rbf_kernel(x: torch.Tensor, x2: torch.Tensor = None, a_vector: torch.Tensor = None, a_scalar: float = 1.0, dim: int = 0):
    if x2 is None:
        return batch_anisotropic_rbf_kernel(x=x, a_vector=a_vector, a_scalar=a_scalar, dim=dim)
    else:
        return instance_anisotropic_rbf_kernel(x1=x, x2=x2, a_vector=a_vector, a_scalar=a_scalar)


def instance_anisotropic_rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, a_vector: torch.Tensor = None, a_scalar: float = 1.0):
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
    return torch.exp(-d @ A @ d.T)


def batch_anisotropic_rbf_kernel(x: torch.Tensor, a_vector: torch.Tensor, a_scalar: float = 1.0, dim: int = 0):
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
    if x2 is None:
        return batch_custom_hybrid_kernel(x=x, kernels=kernels, weights=weights, dim=dim)
    else:
        return instance_custom_hybrid_kernel(x1=x, x2=x2, kernels=kernels, weights=weights)


def instance_custom_hybrid_kernel(x1: torch.Tensor, x2: torch.Tensor, kernels: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], weights: Union[List, Tuple, float] = None):
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

