# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# Statistical Metrics #
#######################

import torch


def pearson_correlation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    assert x1.ndim == 1 and x1.shape == x2.shape

    x1_centered = x1 - x1.mean()
    x2_centered = x2 - x2.mean()

    cov_x1_x2 = torch.dot(x1_centered, x2_centered) / (x1.size(0) - 1)
    var_x1 = torch.dot(x1_centered, x1_centered) / (x1.size(0) - 1)
    var_x2 = torch.dot(x2_centered, x2_centered) / (x2.size(0) - 1)

    pearson_corr = cov_x1_x2 / torch.sqrt(var_x1 * var_x2)

    return pearson_corr


def batch_pearson_correlation(X: torch.Tensor) -> torch.Tensor:
    assert X.ndim == 2
    b, o = X.shape

    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean
    cov_matrix = torch.matmul(X_centered.t(), X_centered) / (b - 1)
    std_devs = X_centered.std(dim=0, unbiased=True)
    std_matrix = torch.outer(std_devs, std_devs)
    pearson_corr_matrix = cov_matrix / std_matrix

    assert pearson_corr_matrix.shape == (o, o)
    return pearson_corr_matrix


def kl_divergence(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    assert x1.ndim == 1 and x1.shape == x2.shape

    x1 = x1 / x1.sum()
    x2 = x2 / x2.sum()

    kl_div = torch.sum(x1 * torch.log(x1 / x2))

    return kl_div


def batch_kl_divergence(X: torch.Tensor) -> torch.Tensor:
    assert X.ndim == 2
    b, o = X.shape

    X = X / X.sum(dim=0, keepdim=True)
    epsilon = 1e-10
    X = X + epsilon

    log_X = torch.log(X)
    kl_div_matrix = (X.unsqueeze(2) * (log_X.unsqueeze(2) - log_X.unsqueeze(1))).sum(dim=0)

    assert kl_div_matrix.shape == (o, o)
    return kl_div_matrix


def rv_coefficient(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    assert x1.ndim == 1 and x1.shape == x2.shape

    x1_centered = x1 - x1.mean()
    x2_centered = x2 - x2.mean()

    cov_x1_x2 = torch.dot(x1_centered, x2_centered) / (x1.size(0) - 1)
    var_x1 = torch.dot(x1_centered, x1_centered) / (x1.size(0) - 1)
    var_x2 = torch.dot(x2_centered, x2_centered) / (x2.size(0) - 1)

    rv_coeff = cov_x1_x2 / torch.sqrt(var_x1 * var_x2)

    return rv_coeff


def batch_rv_coefficient(X: torch.Tensor) -> torch.Tensor:
    assert X.ndim == 2
    b, o = X.shape

    X_centered = X - X.mean(dim=0, keepdim=True)
    C = torch.matmul(X_centered.t(), X_centered) / (b - 1)
    C_reshaped = C.unsqueeze(2)

    numerator = torch.sum(C_reshaped * C_reshaped.transpose(1, 2), dim=0)
    var_diag = torch.diagonal(C, dim1=0, dim2=1)
    denominator = torch.sqrt(torch.outer(var_diag, var_diag)) ** 2

    rv_coeff_matrix = numerator / denominator

    assert rv_coeff_matrix.shape == (o, o)
    return rv_coeff_matrix


def mutual_information(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    assert x1.ndim == 1 and x1.shape == x2.shape

    X = torch.stack((x1, x2), dim=0)
    cov_x1 = torch.var(x1, unbiased=True)
    cov_x2 = torch.var(x2, unbiased=True)
    cov_joint = torch.cov(X)

    det_cov_x1 = cov_x1
    det_cov_x2 = cov_x2
    det_cov_joint = torch.det(cov_joint)

    mutual_info = 0.5 * torch.log(det_cov_x1 * det_cov_x2 / det_cov_joint)

    return mutual_info


def batch_mutual_information(X: torch.Tensor) -> torch.Tensor:
    assert X.ndim == 2
    b, o = X.shape

    X_centered = X - X.mean(dim=0, keepdim=True)

    C = torch.matmul(X_centered.t(), X_centered) / (b - 1)
    variances = torch.diagonal(C)

    cov_ij_squared = C.unsqueeze(2) * C.unsqueeze(1)  # Shape: (m, m, m)
    cov_ij_squared = cov_ij_squared.sum(dim=0)  # Shape: (m, m)

    variances_squared = variances.unsqueeze(0) * variances.unsqueeze(1)  # Shape: (m, m)

    mi_matrix = 0.5 * torch.log(variances_squared / cov_ij_squared)
    mi_matrix[torch.isinf(mi_matrix)] = 0

    assert mi_matrix.shape == (o, o)
    return mi_matrix


