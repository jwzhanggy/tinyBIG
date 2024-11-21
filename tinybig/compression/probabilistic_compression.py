# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############################################
# Naive Probabilistic Compression Functions #
#############################################

from typing import Callable

import torch

from tinybig.module.base_function import function
from tinybig.compression import transformation
from tinybig.config.base_config import config


class naive_probabilistic_compression(transformation):
    r"""
        A probabilistic-based compression class for dimensionality reduction.

        This class compresses input data by sampling features probabilistically based on a specified
        distribution or metric. It supports simple sampling, normalization, and probabilistic weighting.

        Notes
        ----------
        Formally, given a data instance $\mathbf{x} \in {R}^m$, we define the probabilistic compression function based on probabilistic sampling as:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \mathbf{t} \in {R}^d,
            \end{equation}
        $$
        where the output vector $\mathbf{t}$ is conditionally dependent on $\mathbf{x}$ following certain distributions. For example, using a Gaussian distribution:

        $$
            \begin{equation}
            \mathbf{t} | \mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}).
            \end{equation}
        $$
        The dimension $d$ of the output vector $\mathbf{t}$ is a hyper-parameter $d = k$ requiring manual setup.

        Attributes
        ----------
        k : int
            Number of features to retain after compression.
        metric : Callable, optional
            Metric function to apply to the input tensor before sampling. Defaults to None.
        simply_sampling : bool
            If True, performs simple sampling without further processing. Defaults to True.
        with_replacement : bool
            If True, samples features with replacement. Defaults to False.
        require_normalization : bool
            If True, normalizes the input tensor before sampling. Defaults to True.
        log_prob : bool
            If True, returns the logarithm of probabilities for the compressed features. Defaults to False.
        distribution_function : torch.distributions
            Probability distribution function used for sampling. Defaults to a uniform distribution.

        Methods
        -------
        __init__(k, name='probabilistic_compression', simply_sampling=True, distribution_function=None, ...)
            Initializes the probabilistic compression instance.
        calculate_D(m: int)
            Validates and returns the number of features to retain (`k`).
        to_config()
            Converts the current configuration into a dictionary format.
        calculate_weights(x: torch.Tensor)
            Computes sampling weights for the input tensor based on the probability distribution.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Applies probabilistic sampling to compress the input tensor.
    """
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
        """
            The initialization method of the naive probabilistic compression function.

            It initializes the compression instance based on the provided probabilistic distribution function.

            Parameters
            ----------
            k : int
                Number of features to retain after compression.
            name : str, optional
                Name of the transformation. Defaults to 'probabilistic_compression'.
            simply_sampling : bool, optional
                If True, performs simple sampling without further processing. Defaults to True.
            distribution_function : torch.distributions, optional
                Pre-defined probability distribution function for sampling. Defaults to None.
            distribution_function_configs : dict, optional
                Configuration dictionary for initializing the distribution function. Defaults to None.
            metric : Callable, optional
                Metric function to apply to the input tensor before sampling. Defaults to None.
            with_replacement : bool, optional
                If True, samples features with replacement. Defaults to False.
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
        """
            Validates and returns the number of features to retain (`k`).

            This method ensures that the number of features to retain (`k`) is within the valid range
            [0, m], where `m` is the total number of features in the input.

            Parameters
            ----------
            m : int
                Total number of features in the input tensor.

            Returns
            -------
            int
                The number of features to retain (`k`).

            Raises
            ------
            AssertionError
                If `k` is not set or is not within the range [0, m].
        """
        assert self.k is not None and 0 <= self.k <= m
        return self.k

    def to_config(self):
        """
            Converts the current configuration into a dictionary format.

            This method extracts the current configuration of the instance, including the distribution
            function and its parameters, and returns it as a dictionary.

            Returns
            -------
            dict
                A dictionary containing the configuration of the instance, including parameters for the
                distribution function.
        """
        configs = super().to_config()
        configs['function_parameters'].pop('distribution_function')
        if self.distribution_function is not None:
            configs['function_parameters']['distribution_function_configs'] = function.functions_to_configs(self.distribution_function)
        return configs

    def calculate_weights(self, x: torch.Tensor):
        """
            Computes sampling weights for the input tensor based on the probability distribution.

            This method applies the specified distribution function to compute weights for sampling.
            If no distribution function is provided, uniform weights are assigned.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, num_features)`.

            Returns
            -------
            torch.Tensor
                Sampling weights of shape `(batch_size, num_features)`, normalized to sum to 1 along
                the last dimension.
        """
        if self.distribution_function is not None:
            x = torch.exp(self.distribution_function.log_prob(x))
            weights = x/x.sum(dim=-1, keepdim=True)
        else:
            b, m = x.shape
            weights = torch.ones((b, m)) / m
        return weights

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
            Applies probabilistic sampling to compress the input tensor.

            This method processes the input tensor, computes sampling weights, and uses them to select
            features probabilistically based on the specified distribution function or metric.

            Formally, given a data instance $\mathbf{x} \in {R}^m$, we define the probabilistic compression function based on probabilistic sampling as:

            $$
                \begin{equation}
                \kappa(\mathbf{x}) = \mathbf{t} \in {R}^d,
                \end{equation}
            $$
            where the output vector $\mathbf{t}$ is conditionally dependent on $\mathbf{x}$ following certain distributions. For example, using a Gaussian distribution:

            $$
                \begin{equation}
                \mathbf{t} | \mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}).
                \end{equation}
            $$
            The dimension $d$ of the output vector $\mathbf{t}$ is a hyper-parameter $d = k$ requiring manual setup.


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
                Compressed tensor of shape `(batch_size, k)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected `(batch_size, k)`.
        """
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
    """
        A uniform probabilistic compression class.

        This class samples features using a uniform distribution between specified lower and upper bounds.

        Methods
        -------
        __init__(name='naive_uniform_probabilistic_compression', low=0.0, high=1.0, ...)
            Initializes the uniform probabilistic compression instance.

        Parameters
        ----------
        name : str, optional
            Name of the transformation. Defaults to 'naive_uniform_probabilistic_compression'.
        low : float, optional
            Lower bound of the uniform distribution. Defaults to 0.0.
        high : float, optional
            Upper bound of the uniform distribution. Defaults to 1.0.
        require_normalization : bool, optional
            If True, normalizes the input tensor before sampling. Defaults to True.
    """
    def __init__(self, name: str = 'naive_normal_probabilistic_compression', low: float = 0.0, high: float = 1.0, require_normalization: bool = True, *args, **kwargs):
        """
            Initializes the uniform probabilistic compression instance.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'naive_uniform_probabilistic_compression'.
            low : float, optional
                Lower bound of the uniform distribution. Defaults to 0.0.
            high : float, optional
                Upper bound of the uniform distribution. Defaults to 1.0.
            require_normalization : bool, optional
                If True, normalizes the input tensor before sampling. Defaults to True.
        """
        distribution_function = torch.distributions.uniform.Uniform(low=low, high=high)
        super().__init__(name=name, distribution_function=distribution_function, require_normalization=True, *args, **kwargs)



class naive_normal_probabilistic_compression(naive_probabilistic_compression):
    """
        A normal probabilistic compression class.

        This class samples features using a normal distribution with a specified mean and standard deviation.

        Methods
        -------
        __init__(name='naive_normal_probabilistic_compression', mean=0.0, std=1.0, ...)
            Initializes the normal probabilistic compression instance.

        Parameters
        ----------
        name : str, optional
            Name of the transformation. Defaults to 'naive_normal_probabilistic_compression'.
        mean : float, optional
            Mean of the normal distribution. Defaults to 0.0.
        std : float, optional
            Standard deviation of the normal distribution. Defaults to 1.0.
    """
    def __init__(self, name: str = 'naive_normal_probabilistic_compression', mean: float = 0.0, std: float = 1.0, *args, **kwargs):
        """
            Initializes the normal probabilistic compression instance.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'naive_normal_probabilistic_compression'.
            mean : float, optional
                Mean of the normal distribution. Defaults to 0.0.
            std : float, optional
                Standard deviation of the normal distribution. Defaults to 1.0.
        """
        distribution_function = torch.distributions.normal.Normal(torch.tensor([mean]), torch.tensor([std]))
        super().__init__(name=name, distribution_function=distribution_function, *args, **kwargs)


class naive_cauchy_probabilistic_compression(naive_probabilistic_compression):
    """
        A Cauchy probabilistic compression class.

        This class samples features using a Cauchy distribution with a specified location (mean) and scale (spread).

        Methods
        -------
        __init__(name='naive_cauchy_probabilistic_compression', loc=0.0, scale=1.0, ...)
            Initializes the Cauchy probabilistic compression instance.

        Parameters
        ----------
        name : str, optional
            Name of the transformation. Defaults to 'naive_cauchy_probabilistic_compression'.
        loc : float, optional
            Location parameter of the Cauchy distribution (mean). Defaults to 0.0.
        scale : float, optional
            Scale parameter of the Cauchy distribution (spread). Defaults to 1.0.
    """
    def __init__(self, name: str = 'naive_cauchy_probabilistic_compression', loc: float = 0.0, scale: float = 1.0, *args, **kwargs):
        """
            Initializes the Cauchy probabilistic compression instance.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'naive_cauchy_probabilistic_compression'.
            loc : float, optional
                Location parameter of the Cauchy distribution (mean). Defaults to 0.0.
            scale : float, optional
                Scale parameter of the Cauchy distribution (spread). Defaults to 1.0.
        """
        distribution_function = torch.distributions.cauchy.Cauchy(torch.tensor([loc]), torch.tensor([scale]))
        super().__init__(name=name, distribution_function=distribution_function, *args, **kwargs)


class naive_chi2_probabilistic_compression(naive_probabilistic_compression):
    """
        A Chi-squared probabilistic compression class.

        This class samples features using a Chi-squared distribution with a specified degree of freedom (df).

        Methods
        -------
        __init__(name='naive_chi2_probabilistic_compression', df=1.0, require_normalization=True, ...)
            Initializes the Chi-squared probabilistic compression instance.

        Parameters
        ----------
        name : str, optional
            Name of the transformation. Defaults to 'naive_chi2_probabilistic_compression'.
        df : float, optional
            Degrees of freedom for the Chi-squared distribution. Defaults to 1.0.
        require_normalization : bool, optional
            If True, normalizes the input tensor before sampling. Defaults to True.
    """
    def __init__(self, name: str = 'naive_chi2_probabilistic_compression', df: float = 1.0, require_normalization: bool = True,  *args, **kwargs):
        """
            Initializes the Chi-squared probabilistic compression instance.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'naive_chi2_probabilistic_compression'.
            df : float, optional
                Degrees of freedom for the Chi-squared distribution. Defaults to 1.0.
            require_normalization : bool, optional
                If True, normalizes the input tensor before sampling. Defaults to True.
        """
        distribution_function = torch.distributions.chi2.Chi2(df=torch.tensor([df]))
        super().__init__(name=name, distribution_function=distribution_function, require_normalization=True,  *args, **kwargs)


class naive_exponential_probabilistic_compression(naive_probabilistic_compression):
    """
        An Exponential probabilistic compression class.

        This class samples features using an Exponential distribution with a specified rate (lambda).

        Methods
        -------
        __init__(name='naive_exponential_probabilistic_compression', rate=0.5, require_normalization=True, ...)
            Initializes the Exponential probabilistic compression instance.

        Parameters
        ----------
        name : str, optional
            Name of the transformation. Defaults to 'naive_exponential_probabilistic_compression'.
        rate : float, optional
            Rate parameter (lambda) of the Exponential distribution. Defaults to 0.5.
        require_normalization : bool, optional
            If True, normalizes the input tensor before sampling. Defaults to True.
    """
    def __init__(self, name: str = 'naive_exponential_probabilistic_compression', rate: float = 0.5, require_normalization: bool = True,  *args, **kwargs):
        """
            Initializes the Exponential probabilistic compression instance.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'naive_exponential_probabilistic_compression'.
            rate : float, optional
                Rate parameter (lambda) of the Exponential distribution. Defaults to 0.5.
            require_normalization : bool, optional
                If True, normalizes the input tensor before sampling. Defaults to True.
        """
        distribution_function = torch.distributions.exponential.Exponential(rate=torch.tensor([rate]))
        super().__init__(name=name, distribution_function=distribution_function, require_normalization=True,  *args, **kwargs)


class naive_gamma_probabilistic_compression(naive_probabilistic_compression):
    """
        A Gamma probabilistic compression class.

        This class samples features using a Gamma distribution with specified concentration (alpha) and rate (beta).

        Methods
        -------
        __init__(name='naive_gamma_probabilistic_compression', concentration=1.0, rate=1.0, ...)
            Initializes the Gamma probabilistic compression instance.

        Parameters
        ----------
        name : str, optional
            Name of the transformation. Defaults to 'naive_gamma_probabilistic_compression'.
        concentration : float, optional
            Concentration parameter (alpha) of the Gamma distribution. Defaults to 1.0.
        rate : float, optional
            Rate parameter (beta) of the Gamma distribution. Defaults to 1.0.
        require_normalization : bool, optional
            If True, normalizes the input tensor before sampling. Defaults to True.
    """
    def __init__(self, name: str = 'naive_gamma_probabilistic_compression', concentration: float = 1.0, rate: float = 1.0, require_normalization: bool = True, *args, **kwargs):
        """
            Initializes the Gamma probabilistic compression instance.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'naive_gamma_probabilistic_compression'.
            concentration : float, optional
                Concentration parameter (alpha) of the Gamma distribution. Defaults to 1.0.
            rate : float, optional
                Rate parameter (beta) of the Gamma distribution. Defaults to 1.0.
            require_normalization : bool, optional
                If True, normalizes the input tensor before sampling. Defaults to True.
        """
        distribution_function = torch.distributions.gamma.Gamma(torch.tensor([concentration]), torch.tensor([rate]))
        super().__init__(name=name, distribution_function=distribution_function, require_normalization=True, *args, **kwargs)


class naive_laplace_probabilistic_compression(naive_probabilistic_compression):
    """
        A Laplace probabilistic compression class.

        This class samples features using a Laplace distribution with a specified location (mean) and scale (spread).

        Methods
        -------
        __init__(name='naive_laplace_probabilistic_compression', loc=0.0, scale=1.0, ...)
            Initializes the Laplace probabilistic compression instance.

        Parameters
        ----------
        name : str, optional
            Name of the transformation. Defaults to 'naive_laplace_probabilistic_compression'.
        loc : float, optional
            Location parameter of the Laplace distribution (mean). Defaults to 0.0.
        scale : float, optional
            Scale parameter of the Laplace distribution (spread). Defaults to 1.0.
    """
    def __init__(self, name: str = 'naive_laplace_probabilistic_compression', loc: float = 0.0, scale: float = 1.0, *args, **kwargs):
        """
            Initializes the Laplace probabilistic compression instance.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'naive_laplace_probabilistic_compression'.
            loc : float, optional
                Location parameter of the Laplace distribution (mean). Defaults to 0.0.
            scale : float, optional
                Scale parameter of the Laplace distribution (spread). Defaults to 1.0.
        """
        distribution_function = torch.distributions.laplace.Laplace(torch.tensor([loc]), torch.tensor([scale]))
        super().__init__(name=name, distribution_function=distribution_function, *args, **kwargs)

