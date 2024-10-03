# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# Statistical Kernels #
#######################

from typing import Callable, Tuple, List, Union
import torch


def kernel(
    kernel_name: str = 'pearson_correlation',
    x: torch.Tensor = None, x2: torch.Tensor = None,
    *args, **kwargs
):

    if 'batch' in kernel_name:
        assert x is not None and x2 is None
    else:
        assert x is not None and x2 is not None

    match kernel_name:
        case 'pearson_correlation_kernel' | 'pearson_correlation' | 'pearson': return instance_pearson_correlation_kernel(x1=x, x2=x2)
        case 'batch_pearson_correlation_kernel' | 'batch_pearson_correlation' | 'batch_pearson': return batch_pearson_correlation_kernel(x=x, *args, **kwargs)

        case 'kl_divergence_kernel' | 'kl_divergence': return instance_kl_divergence_kernel(x1=x, x2=x2)
        case 'batch_kl_divergence_kernel' | 'batch_kl_divergence': return batch_kl_divergence_kernel(x=x, *args, **kwargs)

        case 'rv_coefficient_kernel' | 'rv_coefficient': return instance_rv_coefficient_kernel(x1=x, x2=x2)
        case 'batch_rv_coefficient_kernel' | 'batch_rv_coefficient': return batch_rv_coefficient_kernel(x=x, *args, **kwargs)

        case 'mutual_information_kernel' | 'mutual_information': return instance_mutual_information_kernel(x1=x, x2=x2)
        case 'batch_mutual_information_kernel' | 'batch_mutual_information': return batch_mutual_information_kernel(x=x, *args, **kwargs)

        case 'custom_hybrid_kernel' | 'custom_hybrid': return instance_custom_hybrid_kernel(x1=x, x2=x2, *args, **kwargs)
        case 'batch_custom_hybrid_kernel' | 'batch_custom_hybrid': return batch_custom_hybrid_kernel(x=x, *args, **kwargs)

        case _: raise ValueError(f'kernel {kernel_name} not supported')


def pearson_correlation_kernel(x: torch.Tensor, x2: torch.Tensor = None, dim: int = 0):
    if x2 is None:
        return batch_pearson_correlation_kernel(x=x, dim=dim)
    else:
        return instance_pearson_correlation_kernel(x1=x, x2=x2)


def instance_pearson_correlation_kernel(x1: torch.Tensor, x2: torch.Tensor):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    x1_centered = x1 - x1.mean()
    x2_centered = x2 - x2.mean()

    cov_x1_x2 = torch.dot(x1_centered, x2_centered) / (x1.size(0) - 1)
    var_x1 = torch.dot(x1_centered, x1_centered) / (x1.size(0) - 1)
    var_x2 = torch.dot(x2_centered, x2_centered) / (x2.size(0) - 1)

    pearson_corr = cov_x1_x2 / torch.sqrt(var_x1 * var_x2)

    return pearson_corr


def batch_pearson_correlation_kernel(x: torch.Tensor, dim: int = 0):
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if dim == 1:
        x = x.T
    b, m = x.shape

    x_mean = torch.mean(x, dim=0, keepdim=True)
    x_centered = x - x_mean
    cov_matrix = torch.matmul(x_centered.t(), x_centered) / (b - 1)
    std_devs = torch.std(x_centered, dim=0, correction=1)
    std_matrix = torch.outer(std_devs, std_devs)
    pearson_corr_matrix = cov_matrix / std_matrix

    assert pearson_corr_matrix.shape == (m, m)
    return pearson_corr_matrix


def kl_divergence_kernel(x: torch.Tensor, x2: torch.Tensor = None, dim: int = 0):
    if x2 is None:
        return batch_kl_divergence_kernel(x=x, dim=dim)
    else:
        return instance_kl_divergence_kernel(x1=x, x2=x2)


def instance_kl_divergence_kernel(x1: torch.Tensor, x2: torch.Tensor):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1...')

    x1 = torch.softmax(x1, dim=-1)
    x2 = torch.softmax(x2, dim=-1)

    kl_div = torch.sum(x1 * torch.log(x1 / x2))

    return kl_div


def batch_kl_divergence_kernel(x: torch.Tensor, dim: int = 0):
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if dim == 1:
        x = x.T
    b, m = x.shape

    x = torch.softmax(x, dim=0)

    log_x = torch.log(x)
    kl_div_matrix = (x.unsqueeze(2) * (log_x.unsqueeze(2) - log_x.unsqueeze(1))).sum(dim=0)

    assert kl_div_matrix.shape == (m, m)
    return kl_div_matrix


def rv_coefficient_kernel(x: torch.Tensor, x2: torch.Tensor = None, dim: int = 0):
    if x2 is None:
        return batch_rv_coefficient_kernel(x=x, dim=dim)
    else:
        return instance_rv_coefficient_kernel(x1=x, x2=x2)


def instance_rv_coefficient_kernel(x1: torch.Tensor, x2: torch.Tensor):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1 and have the same shape.')

    x1_centered = x1 - x1.mean()
    x2_centered = x2 - x2.mean()

    cov_x1_x2 = torch.dot(x1_centered, x2_centered) / (x1.size(0) - 1)
    var_x1 = torch.dot(x1_centered, x1_centered) / (x1.size(0) - 1)
    var_x2 = torch.dot(x2_centered, x2_centered) / (x2.size(0) - 1)

    rv_coeff = cov_x1_x2 / torch.sqrt(var_x1 * var_x2)

    # Clamp to the range [0, 1] to handle numerical precision issues
    rv_coeff = torch.clamp(rv_coeff, min=0.0, max=1.0)

    return rv_coeff


def batch_rv_coefficient_kernel(x: torch.Tensor, dim: int = 0):
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if dim == 1:
        x = x.T
    b, m = x.shape

    # Center the variables
    x_centered = x - torch.mean(x, dim=0, keepdim=True)

    # Covariance matrix for all variables
    cov_matrix = torch.matmul(x_centered.t(), x_centered) / (b - 1)

    # Initialize RV coefficient matrix
    rv_coeff_matrix = torch.zeros(m, m)

    # Compute pairwise RV coefficients
    for i in range(m):
        for j in range(i + 1, m):
            x1_centered = x_centered[:, i]
            x2_centered = x_centered[:, j]

            cov_x1_x2 = torch.dot(x1_centered, x2_centered) / (b - 1)
            var_x1 = torch.dot(x1_centered, x1_centered) / (b - 1)
            var_x2 = torch.dot(x2_centered, x2_centered) / (b - 1)

            # Compute the RV coefficient
            rv_coeff = cov_x1_x2 / torch.sqrt(var_x1 * var_x2)

            # Clamp to the range [0, 1] to handle numerical precision issues
            rv_coeff = torch.clamp(rv_coeff, min=0.0, max=1.0)

            rv_coeff_matrix[i, j] = rv_coeff
            rv_coeff_matrix[j, i] = rv_coeff  # Symmetric entry

    return rv_coeff_matrix



def mutual_information_kernel(x: torch.Tensor, x2: torch.Tensor = None, dim: int = 0):
    if x2 is None:
        return batch_mutual_information_kernel(x=x, dim=dim)
    else:
        return instance_mutual_information_kernel(x1=x, x2=x2)


def instance_mutual_information_kernel(x1: torch.Tensor, x2: torch.Tensor):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
    if x1.ndim != 1 or x1.shape != x2.shape:
        raise ValueError('x1 and x2 must be of dimension 1 and have the same shape.')

    x = torch.stack((x1, x2), dim=0)
    cov_joint = torch.cov(x)  # Compute joint covariance matrix

    # Compute determinant of the covariance matrices
    det_cov_x1 = torch.var(x1, unbiased=True)
    det_cov_x2 = torch.var(x2, unbiased=True)
    det_cov_joint = torch.linalg.det(cov_joint)

    # Mutual information based on the determinants
    mutual_info = 0.5 * torch.log(det_cov_x1 * det_cov_x2 / det_cov_joint)

    return mutual_info


def batch_mutual_information_kernel(x: torch.Tensor, dim: int = 0):
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
    if x.ndim != 2:
        raise ValueError('Input must be a 2D tensor for batch operation.')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if dim == 1:
        x = x.T
    b, m = x.shape

    # Center the variables
    x_centered = x - torch.mean(x, dim=0, keepdim=True)

    # Compute the covariance matrix for all variables
    cov_matrix = torch.matmul(x_centered.t(), x_centered) / (b - 1)

    # Get the variances (diagonal elements of the covariance matrix)
    variances = torch.diagonal(cov_matrix)

    # Prevent small or zero variances (to avoid division by zero)
    epsilon = 1e-10
    variances = torch.clamp(variances, min=epsilon)

    # Prepare mutual information matrix
    mi_matrix = torch.zeros(m, m)

    # Loop over pairs of variables and calculate MI based on covariance matrix
    for i in range(m):
        for j in range(i + 1, m):
            x1 = x[:, i]
            x2 = x[:, j]

            # Stack the two variables and compute joint covariance
            x_stack = torch.stack((x1, x2), dim=0)
            cov_joint = torch.cov(x_stack)

            # Determinants
            det_cov_x1 = torch.var(x1, unbiased=True).clamp(min=epsilon)
            det_cov_x2 = torch.var(x2, unbiased=True).clamp(min=epsilon)
            det_cov_joint = torch.linalg.det(cov_joint).clamp(min=epsilon)

            # Mutual information calculation
            mi_value = 0.5 * torch.log(det_cov_x1 * det_cov_x2 / det_cov_joint)
            mi_matrix[i, j] = mi_value
            mi_matrix[j, i] = mi_value  # Symmetric

    return mi_matrix




def custom_hybrid_kernel(x: torch.Tensor, x2: torch.Tensor = None, kernels: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, weights: Union[List, Tuple, float] = None, dim: int = 0):
    if x2 is None:
        return batch_custom_hybrid_kernel(x=x, kernels=kernels, weights=weights, dim=dim)
    else:
        return instance_custom_hybrid_kernel(x1=x, x2=x2, kernels=kernels, weights=weights)


def instance_custom_hybrid_kernel(x1: torch.Tensor, x2: torch.Tensor, kernels: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], weights: Union[List, Tuple, float] = None):
    if x1 is None or x2 is None or x1.numel() == 0 or x2.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
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

    print(kernels)
    kernel_outputs = [kernel(x1, x2) for kernel in kernels]
    shapes = [output.shape for output in kernel_outputs]
    if len(set(shapes)) != 1:
        raise ValueError("All kernel outputs must have the same shape.")

    weighted_sum = sum(weight * output for weight, output in zip(weights, kernel_outputs))
    return weighted_sum


def batch_custom_hybrid_kernel(x: torch.Tensor, kernels: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], weights: Union[List, Tuple, float] = None, dim: int = 0):
    if x is None or x.numel() == 0:
        raise ValueError("Input tensors must not be None or empty")
    if x.ndim != 2:
        raise ValueError('x must be of dimension 2...')
    if dim not in [0, 1]:
        raise ValueError('dim must be 0 or 1')

    if kernels is None or len(kernels) == 0:
        raise ValueError("At least one kernel function must be provided.")

    if not isinstance(kernels, list):
        kernels = [kernels]

    if weights is None:
        weights = [1 / len(kernels)] * len(kernels)
    elif not isinstance(weights, list):
        weights = [weights]

    if len(kernels) != len(weights):
        raise ValueError("The number of kernels must match the number of weights.")

    kernel_outputs = [kernel(x) for kernel in kernels]

    shapes = [output.shape for output in kernel_outputs]
    if len(set(shapes)) != 1:
        raise ValueError("All kernel outputs must have the same shape.")

    weighted_sum = sum(weight * output for weight, output in zip(weights, kernel_outputs))
    return weighted_sum