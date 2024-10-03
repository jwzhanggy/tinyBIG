# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Low-rank parameter reconciliation functions.

This module contains the low-rank parameter reconciliation functions,
including lorr_reconciliation, hm_reconciliation, lphm_reconciliation, and dual_lphm_reconciliation.
"""

import torch
import torch.nn.functional as F

from tinybig.reconciliation import fabrication


#######################################
# Random Matrix based reconciliations #
#######################################


class random_matrix_adaption_reconciliation(fabrication):

    def __init__(self, name: str = 'random_matrix_adaption_reconciliation', r: int = 2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.A = None
        self.B = None

    def calculate_l(self, n: int, D: int):
        return n + self.r

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        assert w.ndim == 2 and w.numel() == self.calculate_l(n=n, D=D)
        lambda_1, lambda_2 = torch.split(w, [n, self.r], dim=1)

        Lambda_1 = torch.diag(lambda_1.view(-1)).to(device)
        Lambda_2 = torch.diag(lambda_2.view(-1)).to(device)

        if self.A is None or (self.A is not None and self.A.shape != (n, self.r)):
            self.A = torch.randn(n, self.r, device=device)
        if self.B is None or (self.B is not None and self.B.shape != (D, self.r)):
            self.B = torch.randn(D, self.r, device=device)
        assert self.A.shape == (n, self.r) and self.B.shape == (D, self.r)

        W = torch.matmul(torch.matmul(torch.matmul(Lambda_1, self.A), Lambda_2), self.B.t())
        assert W.shape == (n, D)
        return W


class random_matrix_hypernet_reconciliation(fabrication):

    def __init__(self, name='random_matrix_hypernet_reconciliation', r: int = 2, l: int = 64, hidden_dim: int = 128, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.l = l
        self.hidden_dim = hidden_dim

        self.P = None
        self.Q = None
        self.S = None
        self.T = None

    def calculate_l(self, n: int = None, D: int = None):
        assert self.l is not None
        return self.l

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        assert w.ndim == 2 and w.numel() == self.calculate_l(n=n, D=D)

        if self.P is None or (self.P is not None and self.P.shape != (self.l, self.r)):
            self.P = torch.randn(self.l, self.r, device=device)
        if self.Q is None or (self.Q is not None and self.Q.shape != (self.hidden_dim, self.r)):
            self.Q = torch.randn(self.hidden_dim, self.r, device=device)
        assert self.P.shape == (self.l, self.r) and self.Q.shape == (self.hidden_dim, self.r)

        if self.S is None or (self.S is not None and self.S.shape != (self.hidden_dim, self.r)):
            self.S = torch.randn(self.hidden_dim, self.r, device=device)
        if self.T is None or (self.T is not None and self.T.shape != (n*D, self.r)):
            self.T = torch.randn(n*D, self.r, device=device)
        assert self.S.shape == (self.hidden_dim, self.r) and self.T.shape == (n*D, self.r)

        W = torch.matmul(
            torch.matmul(
                F.sigmoid(torch.matmul(torch.matmul(w, self.P), self.Q.t())),
                self.S),
            self.T.t()
        ).view(n, D)

        assert W.shape == (n, D)
        return W

