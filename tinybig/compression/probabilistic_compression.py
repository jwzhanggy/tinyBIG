# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############################################
# Naive Probabilistic Compression Functions #
#############################################

from typing import Callable
import numpy as np

import torch

from tinybig.module.base_function import function
from tinybig.compression import transformation
from tinybig.config.base_config import config


class naive_probabilistic_compression(transformation):
    def __init__(
        self,
        k: int,
        name: str = 'probabilistic_compression',
        simply_sampling: bool = True,
        distribution_function: torch.distributions = None,
        distribution_function_configs: dict = None,
        metric: Callable[[torch.Tensor], torch.Tensor] = None,
        with_replacement: bool = False,
        require_normalization: bool = True,
        log_prob: bool = False,
        *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.k = k
        self.metric = metric
        self.simply_sampling = simply_sampling
        self.with_replacement = with_replacement
        self.require_normalization = require_normalization
        self.log_prob = log_prob

        if self.simply_sampling:
            self. log_prob = False

        if distribution_function is not None:
            self.distribution_function = distribution_function
        elif distribution_function_configs is not None:
            function_class = distribution_function_configs['function_class']
            function_parameters = distribution_function_configs['function_parameters'] if 'function_parameters' in distribution_function_configs else {}
            self.distribution_function = config.get_obj_from_str(function_class)(**function_parameters)
        else:
            self.distribution_function = None

        if self.distribution_function is None:
            self.distribution_function = torch.distributions.uniform.Uniform(low=0.0, high=1.0)

    def calculate_D(self, m: int):
        assert self.k is not None and 0 <= self.k <= m
        return self.k

    def to_config(self):
        configs = super().to_config()
        configs['function_parameters'].pop('distribution_function')
        if self.distribution_function is not None:
            configs['function_parameters']['distribution_function_configs'] = function.functions_to_configs(self.distribution_function)
        return configs

    def calculate_weights(self, x: torch.Tensor):
        if self.distribution_function is not None:
            x = torch.exp(self.distribution_function.log_prob(x))
            weights = x/x.sum(dim=-1, keepdim=True)
        else:
            b, m = x.shape
            weights = torch.ones((b, m)) / m
        return weights

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        data_x = None
        if self.simply_sampling:
            data_x = x.clone()

        if self.metric is not None:
            x = self.metric(x)
        if self.require_normalization:
            x = 0.99 * torch.nn.functional.sigmoid(x) + 0.001

        weights = self.calculate_weights(x)
        sampled_indices = torch.multinomial(weights, self.calculate_D(m=m), replacement=self.with_replacement)
        sampled_indices, _ = torch.sort(sampled_indices, dim=1)

        if self.simply_sampling:
            compression = torch.gather(data_x, 1, sampled_indices)
        else:
            compression = torch.gather(x, 1, sampled_indices)

        if self.log_prob:
            compression = self.distribution_function.log_prob(compression)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class naive_uniform_probabilistic_compression(naive_probabilistic_compression):
    def __init__(self, name: str = 'naive_normal_probabilistic_compression', low: float = 0.0, high: float = 1.0, require_normalization: bool = True, *args, **kwargs):
        distribution_function = torch.distributions.uniform.Uniform(low=low, high=high)
        super().__init__(name=name, distribution_function=distribution_function, require_normalization=True, *args, **kwargs)



class naive_normal_probabilistic_compression(naive_probabilistic_compression):
    def __init__(self, name: str = 'naive_normal_probabilistic_compression', mean: float = 0.0, std: float = 1.0, *args, **kwargs):
        distribution_function = torch.distributions.normal.Normal(torch.tensor([mean]), torch.tensor([std]))
        super().__init__(name=name, distribution_function=distribution_function, *args, **kwargs)


class naive_cauchy_probabilistic_compression(naive_probabilistic_compression):
    def __init__(self, name: str = 'naive_cauchy_probabilistic_compression', loc: float = 0.0, scale: float = 1.0, *args, **kwargs):
        distribution_function = torch.distributions.cauchy.Cauchy(torch.tensor([loc]), torch.tensor([scale]))
        super().__init__(name=name, distribution_function=distribution_function, *args, **kwargs)


class naive_chi2_probabilistic_compression(naive_probabilistic_compression):
    def __init__(self, name: str = 'naive_chi2_probabilistic_compression', df: float = 1.0, require_normalization: bool = True,  *args, **kwargs):
        distribution_function = torch.distributions.chi2.Chi2(df=torch.tensor([df]))
        super().__init__(name=name, distribution_function=distribution_function, require_normalization=True,  *args, **kwargs)


class naive_exponential_probabilistic_compression(naive_probabilistic_compression):
    def __init__(self, name: str = 'naive_exponential_probabilistic_compression', rate: float = 0.5, require_normalization: bool = True,  *args, **kwargs):
        distribution_function = torch.distributions.exponential.Exponential(rate=torch.tensor([rate]))
        super().__init__(name=name, distribution_function=distribution_function, require_normalization=True,  *args, **kwargs)


class naive_gamma_probabilistic_compression(naive_probabilistic_compression):
    def __init__(self, name: str = 'naive_gamma_probabilistic_compression', concentration: float = 1.0, rate: float = 1.0, require_normalization: bool = True, *args, **kwargs):
        distribution_function = torch.distributions.gamma.Gamma(torch.tensor([concentration]), torch.tensor([rate]))
        super().__init__(name=name, distribution_function=distribution_function, require_normalization=True, *args, **kwargs)


class naive_laplace_probabilistic_compression(naive_probabilistic_compression):
    def __init__(self, name: str = 'naive_laplace_probabilistic_compression', loc: float = 0.0, scale: float = 1.0, *args, **kwargs):
        distribution_function = torch.distributions.laplace.Laplace(torch.tensor([loc]), torch.tensor([scale]))
        super().__init__(name=name, distribution_function=distribution_function, *args, **kwargs)

