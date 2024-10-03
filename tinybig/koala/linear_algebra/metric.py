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
    assert x.ndim == 1
    if p == 'nuc':
        raise ValueError(f'the {p}-norm cannot be applied to 1d tensor inputs...')
    else:
        return torch.norm(x, p=p)


def batch_norm(x: torch.Tensor, p: Union[int, float, str, Any], dim: int = 1):
    assert x.ndim == 2 and dim in [0, 1, None]
    if p == 'nuc':
        return torch.norm(x, p='nuc')
    else:
        return torch.norm(x, p=p, dim=dim)


def l1_norm(x: torch.Tensor):
    return norm(x, p=1)


def batch_l1_norm(x: torch.Tensor, dim: int = 1):
    return batch_norm(x, p=1, dim=dim)


def l2_norm(x: torch.Tensor):
    return norm(x, p=2)


def batch_l2_norm(x: torch.Tensor, dim: int = 1):
    return batch_norm(x, p=2, dim=dim)


def sum(x: torch.Tensor):
    assert x.ndim == 1
    return torch.sum(x)


def batch_sum(x: torch.Tensor, dim: int = 1):
    assert x.ndim == 2 and dim in [0, 1]
    return torch.sum(x, dim=dim)


def prod(x: torch.Tensor):
    assert x.ndim == 1
    return torch.prod(x)


def batch_prod(x: torch.Tensor, dim: int = 1):
    assert x.ndim == 2 and dim in [0, 1]
    return torch.prod(x, dim=dim)


def max(x: torch.Tensor):
    assert x.ndim == 1
    return torch.max(x)


def batch_max(x: torch.Tensor, dim: int = 1):
    assert x.ndim == 2 and dim in [0, 1]
    return torch.max(x, dim=dim).values


def min(x: torch.Tensor):
    assert x.ndim == 1
    return torch.min(x)


def batch_min(x: torch.Tensor, dim: int = 1):
    assert x.ndim == 2 and dim in [0, 1]
    return torch.min(x, dim=dim).values


if __name__ == '__main__':

    x = torch.tensor([[1, 2], [3, 4]])
    print(batch_min(x, dim=0), batch_min(x, dim=1), batch_max(x, dim=0), batch_max(x, dim=1))
    y = torch.tensor([1, 2, 3])
    print(min(y), max(y))