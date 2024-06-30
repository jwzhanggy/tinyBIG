# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch
import torch.nn.functional as F

from tinybig.reconciliation import reconciliation


####################
# Lora reconciliation #
####################


class lorr_reconciliation(reconciliation):
    def __init__(self, name='low_rank_reconciliation', r=2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.r = r

    def calculate_l(self, n: int, D: int):
        return self.r * (n + D)

    def __call__(self, n: int, D: int, w: torch.nn.Parameter, r=None, *args, **kwargs):
        '''
        Lorr calculates the "parameter matrix" as W=AB^T, where A has dimension [n, r], B has dimension [D, r]
        :param n: target dimension of the model layer
        :param D: dimension after expansion
        :param w: parameter vector of length l
        :param r: rank of A and B
        :param args:
        :param kwargs:
        :return: product AB^T of dimension [n, D]
        '''
        if r is not None and r != self.r:
            self.r = r
        assert w.dim() == 2 and w.size(1) == self.calculate_l(n=n, D=D)
        A, B = torch.split(w, [self.r*n, self.r*D], dim=1)
        return F.linear(A.view(n, self.r), B.view(D, self.r))


class hm_reconciliation(reconciliation):
    def __init__(self, name='hypercomplex_multiplication_reconciliation', p=2, q=None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p

    def calculate_l(self, n: int, D: int):
        if n % self.p != 0 or D % self.q != 0:
            raise ValueError('The input dimensions {} and {} cannot be divided by parameter p {} and q {}'.format(n, D, self.p, self.q))
        s, t = int(n / self.p), int(D / self.q)
        assert (self.p * self.q * s * t == n * D)
        return self.p * self.q + s * t

    def __call__(self, n: int, D: int, w: torch.nn.Parameter, *args, **kwargs):
        '''
        hypercomplex calculates the "parameter matrix" as W = A xx B, where A has dimension [p, p], B has dimension [s, r]
        :param n:
        :param D:
        :param w:
        :param args:
        :param kwargs:
        :return:
        '''
        assert w.dim() == 2 and w.size(1) == self.calculate_l(n=n, D=D)
        s, t = int(n/self.p), int(D/self.q)
        A, B = torch.split(w, [self.p*self.q, s*t], dim=1)
        return torch.einsum('pq,st->psqt', A, B).view(self.p*s, self.q*t)


class lphm_reconciliation(reconciliation):
    def __init__(self, name='lphm_reconciliation', p=2, q=None, r=2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p
        self.r = r

    def calculate_l(self, n: int, D: int):
        if n % self.p != 0 or D % self.q != 0:
            raise ValueError('The input dimensions {} and {} cannot be divided by parameter p {} and q {}'.format(n, D, self.p, self.q))
        s, t = int(n / self.p), int(D / self.q)
        assert (self.p * self.q * s * t == n * D)
        return self.p * self.q + s * self.r + t * self.r

    def __call__(self, n: int, D: int, w: torch.nn.Parameter, *args, **kwargs):
        assert w.dim() == 2 and w.size(1) == self.calculate_l(n=n, D=D)
        s, t = int(n/self.p), int(D/self.q)
        A, S, T = torch.split(w, [self.p*self.q, s*self.r, t*self.r], dim=1)
        B = F.linear(S.view(s, -1), T.view(t, -1)).view(1, -1)
        return torch.einsum('pq,st->psqt', A, B).view(self.p*s, self.q*t)


class dual_lphm_reconciliation(reconciliation):
    def __init__(self, name='dual_lphm_reconciliation', p=2, q=None, r=2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p
        self.r = r

    def calculate_l(self, n: int, D: int):
        if n % self.p != 0 or D % self.q != 0:
            raise ValueError('The input dimensions {} and {} cannot be divided by parameter p {} and q {}'.format(n, D, self.p, self.q))
        s, t = int(n / self.p), int(D / self.q)
        assert (self.p * self.q * s * t == n * D)
        return self.p * self.r + self.q * self.r + s * self.r + t * self.r

    def __call__(self, n: int, D: int, w: torch.nn.Parameter, *args, **kwargs):
        assert w.dim() == 2 and w.size(1) == self.calculate_l(n=n, D=D)
        s, t = int(n/self.p), int(D/self.q)
        P, Q, S, T = torch.split(w, [self.p*self.r, self.q*self.r, s*self.r, t*self.r], dim=1)
        A = F.linear(P.view(self.p, -1), Q.view(self.q, -1)).view(1, -1)
        B = F.linear(S.view(s, -1), T.view(t, -1)).view(1, -1)
        return torch.einsum('pq,st->psqt', A, B).view(self.p*s, self.q*t)