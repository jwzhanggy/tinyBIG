# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################################
# Combinatorial Probabilistic Compression Functions #
#####################################################

"""
Combinatorial data compression functions.

This module contains the combinatorial data compression functions,
including combinatorial_compression, and combinatorial_probabilistic_compression.
"""

from typing import Callable
import torch

from tinybig.expansion import transformation


class combinatorial_compression(transformation):
    r"""
        A combinatorial compression class for dimensionality reduction.

        This class generates combinations of features from the input tensor up to a specified order (`d`)
        and applies sampling to reduce the dimensionality.

        Notes
        ----------
        Formally, given a data instance vector $\mathbf{x} \in {R}^m$, we can represent the combination of $k$
        selected attributes from $\mathbf{x}$ as $\mathbf{x} \choose k$, for $k \in \{1, 2, \cdots, m\}$.
        It will introduce the combinatorial expansion of the input data instance vector $\mathbf{x}$ as follows:
        $$
            \begin{equation}
            \kappa(\mathbf{x} | k) = \left[ {\mathbf{x} \choose 1}, {\mathbf{x} \choose 2}, \cdots, {\mathbf{x} \choose k} \right].
            \end{equation}
        $$

        Based on this expansion, we can define the combinatorial probabilistic compression function by sampling $d$
        tuples from $\kappa(\mathbf{x} | 1:k)$, treating the tuples as independent ``items'', whose output dimension
        will be $\sum_{i=1}^d k \times i$.

        Furthermore, a corresponding multivariate distribution can also be applied to compute the log-likelihood of the tuples:
        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \log P\left( {\kappa(\mathbf{x} | k) \choose d} | \boldsymbol{\theta} \right) \in {R}^d,
            \end{equation}
        $$
        which reduce the output dimension to be $\sum_{i=1}^d k \times 1$.

        Attributes
        ----------
        k : int
            Number of combinations to retain per order.
        d : int
            Maximum order of combinations to generate.
        metric : Callable, optional
            Metric function to apply to the input tensor before sampling. Defaults to None.
        simply_sampling : bool
            If True, performs simple sampling without further processing. Defaults to True.
        with_replacement : bool
            If True, allows combinations to be generated with replacement. Defaults to False.
        require_normalization : bool
            If True, normalizes the input tensor before sampling. Defaults to True.
        log_prob : bool
            If True, returns the logarithm of probabilities for the compressed features. Defaults to False.
        distribution_functions : dict
            Dictionary of distribution functions for each combination order.

        Methods
        -------
        __init__(name='combinatorial_compression', d=1, k=1, ...)
            Initializes the combinatorial compression instance.
        calculate_D(m: int)
            Computes the total number of features to retain after compression.
        calculate_weights(x: torch.Tensor, r: int)
            Computes sampling weights for the input tensor based on the probability distribution.
        random_combinations(x: torch.Tensor, r: int, *args, **kwargs)
            Generates random combinations of features and samples the resulting combinations.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Applies combinatorial compression to the input tensor.
    """
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
        """
            Initializes the combinatorial compression instance.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'combinatorial_compression'.
            d : int, optional
                Maximum order of combinations to generate. Defaults to 1.
            k : int, optional
                Number of combinations to retain per order. Defaults to 1.
            simply_sampling : bool, optional
                If True, performs simple sampling without further processing. Defaults to True.
            metric : Callable, optional
                Metric function to apply to the input tensor before sampling. Defaults to None.
            with_replacement : bool, optional
                If True, allows combinations to be generated with replacement. Defaults to False.
            require_normalization : bool, optional
                If True, normalizes the input tensor before sampling. Defaults to True.
            log_prob : bool, optional
                If True, returns the logarithm of probabilities for the compressed features. Defaults to False.
            *args : tuple
                Additional positional arguments for the parent `transformation` class.
            **kwargs : dict
                Additional keyword arguments for the parent `transformation` class.
        """
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
        """
            The compression dimension calculation method.

            It computes the total number of features to retain after compression.
            For each combination order, calculates the number of combinations to retain based on `k` and `d`.

            Parameters
            ----------
            m : int
                Total number of features in the input tensor.

            Returns
            -------
            int
                Total number of features to retain after compression.

            Raises
            ------
            AssertionError
                If `d` is not less than 1 or if `k` is not in the range [0, m].
        """
        assert self.d is not None and self.d >= 1
        assert self.k is not None and 0 <= self.k <= m
        if self.simply_sampling:
            return int(sum([self.k * i for i in range(1, self.d+1)]))
        else:
            return int(self.d * self.k)

    def calculate_weights(self, x: torch.Tensor, r: int):
        """
            Computes sampling weights for the input tensor based on the probability distribution.

            This method applies the specified distribution function for the combination order `r`
            to compute weights for sampling.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, num_features)`.
            r : int
                Combination order for which the weights are computed.

            Returns
            -------
            torch.Tensor
                Sampling weights of shape `(batch_size, num_features)`, normalized to sum to 1 along the last dimension.
        """
        if r in self.distribution_functions and self.distribution_functions[r] is not None:
            x = torch.exp(self.distribution_functions[r].log_prob(x))
            weights = x/x.sum(dim=-1, keepdim=True)
        else:
            b, m = x.shape
            weights = torch.ones((b, m)) / m
        return weights

    def random_combinations(self, x: torch.Tensor, r: int, *args, **kwargs):
        """
            Generates random combinations of features and samples the resulting combinations.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, num_features)`.
            r : int
                Order of combinations to generate.
            *args : tuple
                Additional positional arguments for the method.
            **kwargs : dict
                Additional keyword arguments for the method.

            Returns
            -------
            torch.Tensor
                Compressed tensor containing sampled combinations of shape `(batch_size, k)`.
        """
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
        """
            Applies combinatorial compression to the input tensor.

            Combines features up to the specified order (`d`) and samples combinations to reduce dimensionality.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, num_features)`.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for pre- and post-processing.
            **kwargs : dict
                Additional keyword arguments for pre- and post-processing.

            Returns
            -------
            torch.Tensor
                Compressed tensor of shape `(batch_size, total_features)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected dimensions.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        compression = []
        for r in range(1, self.d+1):
            compression.append(self.random_combinations(x=x, r=r, device=device))
        compression = torch.cat(compression, dim=-1).view(b, -1)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class combinatorial_probabilistic_compression(combinatorial_compression):
    r"""
        A combinatorial probabilistic compression class for dimensionality reduction.

        This class extends `combinatorial_compression` by enabling probabilistic sampling
        of feature combinations based on a defined metric or distribution function.

        Notes
        ----------
        Based on the combinatorial compression function, a corresponding multivariate distribution can also be applied to compute the log-likelihood of the tuples:
        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \log P\left( {\kappa(\mathbf{x} | k) \choose d} | \boldsymbol{\theta} \right) \in {R}^d,
            \end{equation}
        $$
        which reduce the output dimension to be $\sum_{i=1}^d k \times 1$.

        Methods
        -------
        __init__(name='combinatorial_probabilistic_compression', d=1, k=1, ...)
            Initializes the combinatorial probabilistic compression instance.
    """
    def __init__(
        self,
        name: str = 'combinatorial_probabilistic_compression',
        d: int = 1, k: int = 1,
        metric: Callable[[torch.Tensor], torch.Tensor] = None,
        with_replacement: bool = False,
        require_normalization: bool = True,
        *args, **kwargs
    ):
        """
            Initializes the combinatorial probabilistic compression instance.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'combinatorial_probabilistic_compression'.
            d : int, optional
                Maximum order of combinations to generate. Defaults to 1.
            k : int, optional
                Number of combinations to retain per order. Defaults to 1.
            metric : Callable, optional
                Metric function to apply to the input tensor before sampling. Defaults to None.
            with_replacement : bool, optional
                If True, allows combinations to be generated with replacement. Defaults to False.
            require_normalization : bool, optional
                If True, normalizes the input tensor before sampling. Defaults to True.
            *args : tuple
                Additional positional arguments for the parent `combinatorial_compression` class.
            **kwargs : dict
                Additional keyword arguments for the parent `combinatorial_compression` class.
        """
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

