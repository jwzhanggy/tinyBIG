# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
# Numerical Metrics #
#####################

import torch
import numpy as np
from typing import Union, Any


def metric(
    x: torch.Tensor,
    metric_name: str,
    *args, **kwargs
):
    """
        Compute a specified metric for the given tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        metric_name : str
            The name of the metric to compute. Supported metrics:
            - "norm", "batch_norm", "l2_norm", "batch_l2_norm", "l1_norm", "batch_l1_norm"
            - "max", "batch_max", "min", "batch_min", "sum", "batch_sum", "prod", "batch_prod"
        *args, **kwargs
            Additional arguments for specific metrics.

        Returns
        -------
        torch.Tensor
            The computed metric.

        Raises
        ------
        AssertionError
            If the input tensor or metric name is None.
        ValueError
            If an unknown metric name is provided.
        """

    assert x is not None and metric_name is not None

    match metric_name:
        case 'norm': return norm(x=x, *args, **kwargs)
        case 'batch_norm': return batch_norm(x=x, *args, **kwargs)
        case 'l2_norm': return l2_norm(x=x)
        case 'batch_l2_norm': return batch_l2_norm(x=x, *args, **kwargs)
        case 'l1_norm': return l1_norm(x=x)
        case 'batch_l1_norm': return batch_l1_norm(x=x, *args, **kwargs)
        case 'max': return max(x=x)
        case 'batch_max': return batch_max(x=x, *args, **kwargs)
        case 'min': return min(x=x)
        case 'batch_min': return batch_min(x=x, *args, **kwargs)
        case 'sum': return sum(x=x)
        case 'batch_sum': return batch_sum(x=x, *args, **kwargs)
        case 'prod': return prod(x=x)
        case 'batch_prod': return batch_prod(x=x, *args, **kwargs)
        case _: raise ValueError(f'Unknown metric name: {metric_name}...')


def norm(x: torch.Tensor, p: Union[int, float, str, Any]):
    """
        Compute the norm of a 1D tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input 1D tensor.
        p : Union[int, float, str, Any]
            The norm order (e.g., 1, 2, "inf").

        Returns
        -------
        torch.Tensor
            The computed norm.

        Raises
        ------
        AssertionError
            If the input tensor is not 1D.
        ValueError
            If the nuclear norm is requested for a 1D tensor.
        """
    assert x.ndim == 1
    if p == 'nuc':
        raise ValueError(f'the {p}-norm cannot be applied to 1d tensor inputs...')
    else:
        return torch.norm(x, p=p)


def batch_norm(x: torch.Tensor, p: Union[int, float, str, Any], dim: int = 1):
    """
        Compute the norm of a batch of tensors along a specified dimension.

        Parameters
        ----------
        x : torch.Tensor
            The input 2D tensor.
        p : Union[int, float, str, Any]
            The norm order (e.g., 1, 2, "nuc").
        dim : int, optional
            The dimension along which to compute the norm. Default is 1.

        Returns
        -------
        torch.Tensor
            The computed norms.

        Raises
        ------
        AssertionError
            If the input tensor is not 2D or the dimension is invalid.
        """
    assert x.ndim == 2 and dim in [0, 1, None]
    if p == 'nuc':
        return torch.norm(x, p='nuc')
    else:
        return torch.norm(x, p=p, dim=dim)


def l1_norm(x: torch.Tensor):
    """
        Compute the L1 norm of a 1D tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input 1D tensor.

        Returns
        -------
        torch.Tensor
            The L1 norm.
        """
    return norm(x, p=1)


def batch_l1_norm(x: torch.Tensor, dim: int = 1):
    """
        Compute the L1 norm of a batch of tensors along a specified dimension.

        Parameters
        ----------
        x : torch.Tensor
            The input 2D tensor.
        dim : int, optional
            The dimension along which to compute the L1 norm. Default is 1.

        Returns
        -------
        torch.Tensor
            The L1 norms.
        """
    return batch_norm(x, p=1, dim=dim)


def l2_norm(x: torch.Tensor):
    """
        Compute the L2 norm of a 1D tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input 1D tensor.

        Returns
        -------
        torch.Tensor
            The L2 norm.
        """
    return norm(x, p=2)


def batch_l2_norm(x: torch.Tensor, dim: int = 1):
    """
        Compute the L2 norm of a batch of tensors along a specified dimension.

        Parameters
        ----------
        x : torch.Tensor
            The input 2D tensor.
        dim : int, optional
            The dimension along which to compute the L2 norm. Default is 1.

        Returns
        -------
        torch.Tensor
            The L2 norms.
        """
    return batch_norm(x, p=2, dim=dim)


def sum(x: torch.Tensor):
    """
        Compute the sum of elements in a 1D tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input 1D tensor.

        Returns
        -------
        torch.Tensor
            The sum of elements.
        """
    assert x.ndim == 1
    return torch.sum(x)


def batch_sum(x: torch.Tensor, dim: int = 1):
    """
        Compute the sum of elements along a specified dimension in a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor
            The input 2D tensor.
        dim : int, optional
            The dimension along which to compute the sum. Default is 1.

        Returns
        -------
        torch.Tensor
            The sums.
        """
    assert x.ndim == 2 and dim in [0, 1]
    return torch.sum(x, dim=dim)


def prod(x: torch.Tensor):
    """
        Compute the product of elements in a 1D tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input 1D tensor.

        Returns
        -------
        torch.Tensor
            The product of elements.
        """
    assert x.ndim == 1
    return torch.prod(x)


def batch_prod(x: torch.Tensor, dim: int = 1):
    """
        Compute the product of elements along a specified dimension in a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor
            The input 2D tensor.
        dim : int, optional
            The dimension along which to compute the product. Default is 1.

        Returns
        -------
        torch.Tensor
            The products.
        """
    assert x.ndim == 2 and dim in [0, 1]
    return torch.prod(x, dim=dim)


def max(x: torch.Tensor):
    """
        Compute the maximum value in a 1D tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input 1D tensor.

        Returns
        -------
        torch.Tensor
            The maximum value.
        """
    assert x.ndim == 1
    return torch.max(x)


def batch_max(x: torch.Tensor, dim: int = 1):
    """
        Compute the maximum values along a specified dimension in a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor
            The input 2D tensor.
        dim : int, optional
            The dimension along which to compute the maximum values. Default is 1.

        Returns
        -------
        torch.Tensor
            The maximum values.
        """
    assert x.ndim == 2 and dim in [0, 1]
    return torch.max(x, dim=dim).values


def min(x: torch.Tensor):
    """
        Compute the minimum value in a 1D tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input 1D tensor.

        Returns
        -------
        torch.Tensor
            The minimum value.
        """
    assert x.ndim == 1
    return torch.min(x)


def batch_min(x: torch.Tensor, dim: int = 1):
    """
        Compute the minimum values along a specified dimension in a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor
            The input 2D tensor.
        dim : int, optional
            The dimension along which to compute the minimum values. Default is 1.

        Returns
        -------
        torch.Tensor
            The minimum values.
        """
    assert x.ndim == 2 and dim in [0, 1]
    return torch.min(x, dim=dim).values

