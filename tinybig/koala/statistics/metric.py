# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# Statistical Metrics #
#######################

import torch


def metric(
    metric_name: str,
    x: torch.Tensor,
    *args, **kwargs
):
    """
    Computes various statistical metrics based on the specified metric name.

    Parameters
    ----------
    metric_name : str
        The name of the metric to compute. Options include 'mean', 'variance',
        'entropy', 'skewness', etc.
    x : torch.Tensor
        The input tensor.

    Returns
    -------
    torch.Tensor
        The computed metric.

    Raises
    ------
    ValueError
        If the metric name is not recognized or the input tensor is invalid.
    """
    assert x is not None and metric_name is not None
    match metric_name:
        case 'mean': return mean(x=x)
        case 'batch_mean': return batch_mean(x=x, *args, **kwargs)
        case 'weighted_mean' | 'wmean': return weighted_mean(x=x, *args, **kwargs)
        case 'batch_weighted_mean' | 'batch_wmean': return batch_weighted_mean(x=x, *args, **kwargs)
        case 'geometric_mean' | 'gmean': return geometric_mean(x=x)
        case 'batch_geometric_mean' | 'batch_gmean': return batch_geometric_mean(x=x, *args, **kwargs)
        case 'harmonic_mean' | 'hmean': return harmonic_mean(x=x, *args, **kwargs)
        case 'batch_harmonic_mean' | 'batch_hmean': return batch_harmonic_mean(x=x, *args, **kwargs)
        case 'median': return median(x=x)
        case 'batch_median': return batch_median(x=x, *args, **kwargs)
        case 'mode': return mode(x=x)
        case 'batch_mode': return batch_mode(x=x, *args, **kwargs)
        case 'entropy': return entropy(x=x)
        case 'batch_entropy': return batch_entropy(x=x, *args, **kwargs)
        case 'variance' | 'var': return variance(x=x)
        case 'batch_variance' | 'batch_var': return batch_variance(x=x, *args, **kwargs)
        case 'std' | 'standard_deviation': return std(x=x)
        case 'batch_std' | 'batch_standard_deviation': return batch_std(x=x, *args, **kwargs)
        case 'skewness' | 'skew': return skewness(x=x)
        case 'batch_skewness' | 'batch_skew': return batch_skewness(x=x, *args, **kwargs)
        case _: raise ValueError(f'Unknown metric name: {metric_name}...')


def mean(x: torch.Tensor):
    """
    Computes the mean of a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor.

    Returns
    -------
    torch.Tensor
        The mean value.
    """
    assert x.ndim == 1
    return torch.mean(x)


def batch_mean(x: torch.Tensor, dim: int = 1):
    """
    Computes the mean along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor.
    dim : int
        The dimension along which to compute the mean (0 or 1).

    Returns
    -------
    torch.Tensor
        The mean values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1]
    return torch.mean(x, dim=dim)


def weighted_mean(x: torch.Tensor, weights: torch.Tensor):
    """
    Computes the weighted mean of a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor.
    weights : torch.Tensor
        The weights for each element in `x`.

    Returns
    -------
    torch.Tensor
        The weighted mean value.
    """
    assert x.ndim == 1 and x.shape == weights.shape
    weighted_sum = torch.sum(x * weights)
    sum_weights = torch.sum(weights)
    return weighted_sum / sum_weights


def batch_weighted_mean(x: torch.Tensor, weights: torch.Tensor, dim: int = 1):
    """
    Computes the weighted mean along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor.
    weights : torch.Tensor
        The weights for each element in the specified dimension.
    dim : int
        The dimension along which to compute the weighted mean (0 or 1).

    Returns
    -------
    torch.Tensor
        The weighted mean values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1] and x.shape[1] == weights.shape[0]

    weights = weights.unsqueeze(0) if dim == 1 else weights.unsqueeze(1)
    weighted_sum = torch.sum(x * weights, dim=dim)
    sum_weights = torch.sum(weights, dim=dim)

    return weighted_sum / sum_weights


def geometric_mean(x: torch.Tensor):
    """
    Computes the geometric mean of a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor. All elements must be positive.

    Returns
    -------
    torch.Tensor
        The geometric mean value.
    """
    assert x.ndim == 1 and torch.all(x > 0)
    log_x = torch.log(x)
    mean_log_x = torch.mean(log_x)
    return torch.exp(mean_log_x)


def batch_geometric_mean(x: torch.Tensor, dim: int = 1):
    """
    Computes the geometric mean along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor. All elements must be positive.
    dim : int
        The dimension along which to compute the geometric mean (0 or 1).

    Returns
    -------
    torch.Tensor
        The geometric mean values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1] and torch.all(x > 0)

    log_x = torch.log(x)
    mean_log_x = torch.mean(log_x, dim=dim)
    return torch.exp(mean_log_x)


def harmonic_mean(x: torch.Tensor, weights=None):
    """
    Computes the harmonic mean of a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor. All elements must be positive.
    weights : torch.Tensor, optional
        The weights for each element in `x`.

    Returns
    -------
    torch.Tensor
        The harmonic mean value.
    """
    assert x.ndim == 1 and torch.all(x > 0)

    if weights is None:
        return x.numel() / torch.sum(1 / x)
    else:
        assert weights.ndim == 1 and x.shape == weights.shape
        weighted_sum = torch.sum(weights)
        weighted_reciprocal_sum = torch.sum(weights / x)
        return weighted_sum / weighted_reciprocal_sum


def batch_harmonic_mean(x: torch.Tensor, weights=None, dim: int = 1):
    """
    Computes the harmonic mean along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor. All elements must be positive.
    weights : torch.Tensor, optional
        The weights for each element in the specified dimension.
    dim : int
        The dimension along which to compute the harmonic mean (0 or 1).

    Returns
    -------
    torch.Tensor
        The harmonic mean values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1] and torch.all(x > 0)

    if weights is not None:
        assert weights.ndim == 1 and x.shape[dim] == weights.shape[0]

    if weights is None:
        reciprocal_sum = torch.sum(1 / x, dim=dim)
        return x.shape[dim] / reciprocal_sum
    else:
        weighted_reciprocal_sum = torch.sum(weights / x, dim=dim)
        return torch.sum(weights, dim=dim) / weighted_reciprocal_sum


def median(x: torch.Tensor):
    """
    Computes the median of a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor.

    Returns
    -------
    torch.Tensor
        The median value.
    """
    assert x.ndim == 1
    return torch.median(x)


def batch_median(x: torch.Tensor, dim: int = 1):
    """
    Computes the median along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor.
    dim : int
        The dimension along which to compute the median (0 or 1).

    Returns
    -------
    torch.Tensor
        The median values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1]
    return torch.median(x, dim=dim).values


def mode(x: torch.Tensor):
    """
    Computes the mode of a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor.

    Returns
    -------
    torch.Tensor
        The mode value.
    """
    assert x.ndim == 1
    return torch.mode(x)


def batch_mode(x: torch.Tensor, dim: int = 1):
    """
    Computes the mode along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor.
    dim : int
        The dimension along which to compute the mode (0 or 1).

    Returns
    -------
    torch.Tensor
        The mode values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1]
    return torch.mode(x, dim=dim)


def entropy(x: torch.Tensor):
    """
    Computes the entropy of a probability distribution represented by a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor. Values must sum to 1.

    Returns
    -------
    torch.Tensor
        The entropy value.
    """
    assert x.ndim == 1 and torch.all(x >= 0) and torch.isclose(torch.sum(x), torch.tensor(1.0)), "The tensor values must sum to 1."
    entropy_value = -torch.sum(x * torch.log(x + 1e-12))
    return entropy_value


def batch_entropy(x: torch.Tensor, dim: int = 1):
    """
    Computes the entropy along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor. Values in each dimension must sum to 1.
    dim : int
        The dimension along which to compute the entropy (0 or 1).

    Returns
    -------
    torch.Tensor
        The entropy values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1]
    assert torch.all(x >= 0) and torch.allclose(torch.sum(x, dim=dim), torch.tensor(1.0))
    entropy_values = -torch.sum((x+1e-12) * torch.log(x), dim=dim)
    return entropy_values


def variance(x: torch.Tensor):
    """
    Computes the variance of a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor.

    Returns
    -------
    torch.Tensor
        The variance value.
    """
    assert x.ndim == 1
    return torch.var(x)


def batch_variance(x: torch.Tensor, dim: int = 1):
    """
    Computes the variance along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor.
    dim : int
        The dimension along which to compute the variance (0 or 1).

    Returns
    -------
    torch.Tensor
        The variance values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1]
    return torch.var(x, dim=dim)


def std(x: torch.Tensor):
    """
    Computes the standard deviation of a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor.

    Returns
    -------
    torch.Tensor
        The standard deviation value.
    """
    assert x.ndim == 1
    return torch.std(x)


def batch_std(x: torch.Tensor, dim: int = 1):
    """
    Computes the standard deviation along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor.
    dim : int
        The dimension along which to compute the standard deviation (0 or 1).

    Returns
    -------
    torch.Tensor
        The standard deviation values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1]
    return torch.std(x, dim=dim)


def skewness(x: torch.Tensor):
    """
    Computes the skewness of a 1D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 1D tensor.

    Returns
    -------
    torch.Tensor
        The skewness value.
    """
    assert x.ndim == 1
    skewness_value = torch.mean((x - torch.mean(x)) ** 3) / (torch.std(x, unbiased=False) ** 3 + 1e-12)
    return skewness_value


def batch_skewness(x: torch.Tensor, dim: int = 1):
    """
    Computes the skewness along a specified dimension for a 2D tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input 2D tensor.
    dim : int
        The dimension along which to compute the skewness (0 or 1).

    Returns
    -------
    torch.Tensor
        The skewness values along the specified dimension.
    """
    assert x.ndim == 2 and dim in [0, 1]
    skewness_value = torch.mean((x - torch.mean(x, dim=dim, keepdim=True)) ** 3, dim=dim) / (torch.std(x, dim=dim, unbiased=False, keepdim=True) ** 3 + 1e-12)
    return skewness_value.squeeze(dim)
