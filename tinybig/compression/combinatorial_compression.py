# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################################
# Combinatorial Probabilistic Compression Functions #
#####################################################

from typing import Callable
import torch

from tinybig.expansion import transformation


class combinatorial_compression(transformation):

    def __init__(
        self,
        name: str = 'combinatorial_compression',
        d: int = 1, k: int = 1,
        simply_sampling: bool = True,
        metric: Callable[[torch.Tensor], torch.Tensor] = None,
        with_replacement: bool = False,
        require_normalization: bool = True,
        log_prob: bool = False,
        *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.k = k
        self.d = d
        self.metric = metric
        self.simply_sampling = simply_sampling
        self.with_replacement = with_replacement
        self.require_normalization = require_normalization
        self.log_prob = log_prob

        if self.simply_sampling:
            self. log_prob = False

        self.distribution_functions = {}
        for r in range(1, self.d+1):
            self.distribution_functions[r] = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.zeros(r), covariance_matrix=torch.eye(r)
            )

    def calculate_D(self, m: int):
        assert self.d is not None and self.d >= 1
        assert self.k is not None and 0 <= self.k <= m
        if self.simply_sampling:
            return int(sum([self.k * i for i in range(1, self.d+1)]))
        else:
            return int(self.d * self.k)

    def calculate_weights(self, x: torch.Tensor, r: int):
        if r in self.distribution_functions and self.distribution_functions[r] is not None:
            x = torch.exp(self.distribution_functions[r].log_prob(x))
            weights = x/x.sum(dim=-1, keepdim=True)
        else:
            b, m = x.shape
            weights = torch.ones((b, m)) / m
        return weights

    def random_combinations(self, x: torch.Tensor, r: int, *args, **kwargs):
        b, m = x.shape
        assert x.ndim == 2 and 0 < r <= x.shape[1]

        comb = []
        for i in range(x.size(0)):
            comb.append(torch.combinations(input=x[i, :], r=r, with_replacement=self.with_replacement))
        x = torch.stack(comb, dim=0)

        data_x = None
        if self.simply_sampling:
            data_x = x.clone()

        if self.metric is not None:
            x = self.metric(x)
        if self.require_normalization:
            x = 0.99 * torch.nn.functional.sigmoid(x) + 0.001

        weights = self.calculate_weights(x=x, r=r)
        sampled_indices = torch.multinomial(weights, self.k, replacement=self.with_replacement)
        sampled_indices, _ = torch.sort(sampled_indices, dim=1)

        if self.simply_sampling:
            compression = data_x[torch.arange(data_x.size(0)).unsqueeze(1), sampled_indices]
        else:
            compression = x[torch.arange(x.size(0)).unsqueeze(1), sampled_indices]

        if self.log_prob:
            compression = self.distribution_functions[r].log_prob(compression)

        return compression

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        compression = []
        for r in range(1, self.d+1):
            compression.append(self.random_combinations(x=x, r=r, device=device))
        compression = torch.cat(compression, dim=-1).view(b, -1)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class combinatorial_probabilistic_compression(combinatorial_compression):
    def __init__(
        self,
        name: str = 'combinatorial_probabilistic_compression',
        d: int = 1, k: int = 1,
        metric: Callable[[torch.Tensor], torch.Tensor] = None,
        with_replacement: bool = False,
        require_normalization: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            name=name,
            d=d, k=k,
            metric=metric,
            simply_sampling=False,
            log_prob=True,
            with_replacement=with_replacement,
            require_normalization=require_normalization,
            *args, **kwargs
        )