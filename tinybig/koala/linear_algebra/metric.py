# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# Statistical Metrics #
#######################

import torch

from typing import Union, Any


def inner_product(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    assert x1.ndim == 1 and x1.shape == x2.shape

    dot_product = torch.dot(x1, x2)

    return dot_product


def batch_inner_product(X: torch.Tensor, centered: bool = False) -> torch.Tensor:
    assert X.ndim == 2
    b, o = X.shape

    if centered:
        X = X - X.mean(dim=0, keepdim=True)
    inner_product_matrix = torch.matmul(X.T, X)

    assert inner_product_matrix.shape == (o, o)
    return inner_product_matrix


def cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    assert x1.ndim == 1 and x1.shape == x2.shape

    dot_product = torch.dot(x1, x2)
    norm_x1 = torch.norm(x1, p=2)
    norm_x2 = torch.norm(x2, p=2)

    if norm_x1 == 0 or norm_x2 == 0:
        return torch.tensor(0.0)
    else:
        cosine_sim = dot_product / (norm_x1 * norm_x2)
        return cosine_sim


def batch_cosine_similarity(X: torch.Tensor, centered: bool = False) -> torch.Tensor:
    assert X.ndim == 2
    b, o = X.shape

    if centered:
        X = X - X.mean(dim=0, keepdim=True)
    X_norm = torch.norm(X, p=2, dim=0, keepdim=True)
    X_norm[X_norm == 0] = 1.0
    X_normalized = X / X_norm
    similarity_matrix = torch.matmul(X_normalized.T, X_normalized)

    assert similarity_matrix.shape == (o, o)
    return similarity_matrix


def minkowski_distance(x1: torch.Tensor, x2: torch.Tensor, p: Union[int, float, str, Any]) -> torch.Tensor:
    assert x1.ndim == 1 and x1.shape == x2.shape

    distance = torch.norm(x1 - x2, p=p)

    return distance


def batch_minkowski_distance(X: torch.Tensor, p: Union[int, float, str, Any], centered: bool = False) -> torch.Tensor:
    assert X.ndim == 2
    b, o = X.shape

    if centered:
        X = X - X.mean(dim=0, keepdim=True)
    X_expanded_1 = X.unsqueeze(2)
    X_expanded_2 = X.unsqueeze(1)
    distance_matrix = torch.norm(X_expanded_1 - X_expanded_2, p=p, dim=0)

    assert distance_matrix.shape == (o, o)
    return distance_matrix


def manhattan_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return minkowski_distance(x1=x1, x2=x2, p=1)


def batch_manhattan_distance(X: torch.Tensor, centered: bool = False) -> torch.Tensor:
    return batch_minkowski_distance(X=X, p=1, centered=centered)


def euclidean_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return minkowski_distance(x1=x1, x2=x2, p=2)


def batch_euclidean_distance(X: torch.Tensor, centered: bool = False) -> torch.Tensor:
    return batch_minkowski_distance(X=X, p=2, centered=centered)


def chebyshev_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return minkowski_distance(x1=x1, x2=x2, p=torch.inf)


def batch_chebyshev_distance(X: torch.Tensor, centered: bool = False) -> torch.Tensor:
    return batch_minkowski_distance(X=X, p=torch.inf, centered=centered)


def canberra_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    assert x1.ndim == 1 and x1.shape == x2.shape

    numerator = torch.abs(x1 - x2)
    denominator = torch.abs(x1) + torch.abs(x2)
    canberra_dist = torch.sum(numerator / (denominator + 1e-10))

    return canberra_dist


def batch_canberra_distance(X: torch.Tensor) -> torch.Tensor:
    assert X.ndim == 2
    b, o = X.shape

    X_expanded_1 = X.unsqueeze(2)
    X_expanded_2 = X.unsqueeze(1)

    numerator = torch.abs(X_expanded_1 - X_expanded_2)
    denominator = torch.abs(X_expanded_1) + torch.abs(X_expanded_2)
    canberra_dist_matrix = torch.sum(numerator / (denominator + 1e-10), dim=0)

    assert canberra_dist_matrix.shape == (o, o)
    return canberra_dist_matrix


