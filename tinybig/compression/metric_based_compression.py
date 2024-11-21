# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################
# Metric based Compression Function #
#####################################

import torch

from typing import Callable

from tinybig.compression import transformation
from tinybig.koala.statistics import batch_mean, batch_weighted_mean, batch_std, batch_mode, batch_median, batch_entropy, batch_variance, batch_skewness, batch_harmonic_mean, batch_geometric_mean
from tinybig.koala.linear_algebra import batch_sum, batch_norm, batch_prod, batch_min, batch_max, batch_l1_norm, batch_l2_norm


class metric_compression(transformation):
    r"""
        The metric based compression function.

        It performs the data compression based on provided compression metric, which can be min, max, mean, prod, etc.

        ...

        Notes
        ----------
        Formally, given a data instance $\mathbf{x} \in R^m$ and a provided metric $\phi: {R}^m \to {R}^{d_{\phi}}$,
        which transforms it into a dense representation of length $d_{\phi}$,
        we can represent the metric based compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \phi(\mathbf{x}) \in {R}^{d},
            \end{equation}
        $$

        where the compression output vector dimension is $d = d_{\phi}$. For the majority of metric $\phi$ studied in this project,
        the output is typically a scalar, i.e., the dimension $d_{\phi} = 1$.

        Attributes
        ----------
        metric: Callable[[torch.Tensor], torch.Tensor]
            The metric compression metric.
        name: str, default = 'metric_compression'
            Name of the compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the metric compression function.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method to define the compression function.

    """

    def __init__(self, metric: Callable[[torch.Tensor], torch.Tensor], name: str = 'metric_compression', *args, **kwargs):
        """
            The initialization method of the metric based compression function.

            It initializes the metric compression function based on the provided metric mapping.

            Parameters
            ----------
            metric: Callable[[torch.Tensor], torch.Tensor]
                The metric based compression metric.
            name: str, default = 'metric_compression'
                Name of the compression function.

            Returns
            ----------
            transformation
                The metric based compression function.
        """
        super().__init__(name=name, *args, **kwargs)
        self.metric = metric

    def calculate_D(self, m: int):
        r"""
            The metric compression dimension calculation method.

            The compression output vector dimension is $d = d_{\phi}$.
            For the majority of metric $\phi$ studied in this project,
            the output is typically a scalar, i.e., the dimension $d_{\phi} = 1$.

            Parameters
            ----------
            m: int
                The dimension of the input space.

            Returns
            -------
            int
                The dimension of the metric compression space.
        """
        return 1

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
            The forward method of the metric compression function.

            It performs the metric based data compression of the input data and returns the compression result.

            Formally, given a data instance $\mathbf{x} \in R^m$ and a provided metric $\phi: {R}^m \to {R}^{d_{\phi}}$,
            which transforms it into a dense representation of length $d_{\phi}$,
            we can represent the metric based compression function as follows:

            $$
                \begin{equation}
                \kappa(\mathbf{x}) = \phi(\mathbf{x}) \in {R}^{d},
                \end{equation}
            $$

            where the compression output vector dimension is $d = d_{\phi}$.

            Parameters
            ----------
            x: torch.Tensor
                The input data vector.
            device: str, default = 'cpu'
                The device of the input data vector.

            Returns
            -------
            torch.Tensor
                The compression result.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        assert self.metric is not None
        compression = self.metric(x).unsqueeze(1)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class max_compression(metric_compression):
    r"""
        The max metric based compression function.

        It performs the data compression based on provided max metric.

        ...

        Notes
        ----------
        Formally, given a data instance $\mathbf{x} \in R^m$ and a provided metric $\phi: {R}^m \to {R}^{d_{\phi}}$,
        which transforms it into a dense representation of length $d_{\phi}$,
        we can represent the max metric based compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = max(\mathbf{x}) \in {R}^{d}.
            \end{equation}
        $$

        For the max metric studied in this project, the output is typically a scalar, i.e., the dimension $d = d_{\phi} = 1$.

        Attributes
        ----------
        metric: Callable[[torch.Tensor], torch.Tensor]
            The metric compression metric.
        name: str, default = 'max_compression'
            Name of the max compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the max compression function.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method to define the compression function.

    """
    def __init__(self, name: str = 'max_compression', *args, **kwargs):
        """
            The initialization method of the max metric based compression function.

            It initializes the compression function based on the provided max metric.

            Parameters
            ----------
            name: str, default = 'max_compression'
                Name of the compression function.

            Returns
            ----------
            transformation
                The max metric based compression function.
        """
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class min_compression(metric_compression):
    r"""
        The min metric based compression function.

        It performs the data compression based on provided min metric.

        ...

        Notes
        ----------
        Formally, given a data instance $\mathbf{x} \in R^m$ and a provided metric $\phi: {R}^m \to {R}^{d_{\phi}}$,
        which transforms it into a dense representation of length $d_{\phi}$,
        we can represent the min metric based compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = min(\mathbf{x}) \in {R}^{d}.
            \end{equation}
        $$

        For the min metric studied in this project, the output is typically a scalar, i.e., the dimension $d = d_{\phi} = 1$.

        Attributes
        ----------
        metric: Callable[[torch.Tensor], torch.Tensor]
            The metric compression metric.
        name: str, default = 'min_compression'
            Name of the min compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the min compression function.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method to define the compression function.

    """
    def __init__(self, name: str = 'min_compression', *args, **kwargs):
        """
            The initialization method of the min metric based compression function.

            It initializes the compression function based on the provided min metric.

            Parameters
            ----------
            name: str, default = 'min_compression'
                Name of the compression function.

            Returns
            ----------
            transformation
                The min metric based compression function.
        """
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class sum_compression(metric_compression):
    r"""
        The sum metric based compression function.

        It performs the data compression based on provided sum metric.

        ...

        Notes
        ----------
        Formally, given a data instance $\mathbf{x} \in R^m$ and a provided metric $\phi: {R}^m \to {R}^{d_{\phi}}$,
        which transforms it into a dense representation of length $d_{\phi}$,
        we can represent the sum metric based compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = sum(\mathbf{x}) \in {R}^{d}.
            \end{equation}
        $$

        For the sum metric studied in this project, the output is typically a scalar, i.e., the dimension $d = d_{\phi} = 1$.

        Attributes
        ----------
        metric: Callable[[torch.Tensor], torch.Tensor]
            The metric compression metric.
        name: str, default = 'sum_compression'
            Name of the sum compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the sum compression function.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method to define the compression function.

    """
    def __init__(self, name: str = 'sum_compression', *args, **kwargs):
        """
            The initialization method of the sum metric based compression function.

            It initializes the compression function based on the provided sum metric.

            Parameters
            ----------
            name: str, default = 'sum_compression'
                Name of the compression function.

            Returns
            ----------
            transformation
                The sum metric based compression function.
        """
        super().__init__(name=name, metric=batch_sum, *args, **kwargs)


class mean_compression(metric_compression):
    r"""
        The mean metric based compression function.

        It performs the data compression based on provided mean metric.

        ...

        Notes
        ----------
        Formally, given a data instance $\mathbf{x} \in R^m$ and a provided metric $\phi: {R}^m \to {R}^{d_{\phi}}$,
        which transforms it into a dense representation of length $d_{\phi}$,
        we can represent the mean metric based compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = mean(\mathbf{x}) \in {R}^{d}.
            \end{equation}
        $$

        For the mean metric studied in this project, the output is typically a scalar, i.e., the dimension $d = d_{\phi} = 1$.

        Attributes
        ----------
        metric: Callable[[torch.Tensor], torch.Tensor]
            The metric compression metric.
        name: str, default = 'mean_compression'
            Name of the mean compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the mean compression function.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method to define the compression function.

    """
    def __init__(self, name: str = 'mean_compression', *args, **kwargs):
        """
            The initialization method of the mean metric based compression function.

            It initializes the compression function based on the provided mean metric.

            Parameters
            ----------
            name: str, default = 'mean_compression'
                Name of the compression function.

            Returns
            ----------
            transformation
                The mean metric based compression function.
        """
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)


class prod_compression(metric_compression):
    r"""
        The prod metric based compression function.

        It performs the data compression based on provided prod metric.

        ...

        Notes
        ----------
        Formally, given a data instance $\mathbf{x} \in R^m$ and a provided metric $\phi: {R}^m \to {R}^{d_{\phi}}$,
        which transforms it into a dense representation of length $d_{\phi}$,
        we can represent the prod metric based compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = prod(\mathbf{x}) \in {R}^{d}.
            \end{equation}
        $$

        For the prod metric studied in this project, the output is typically a scalar, i.e., the dimension $d = d_{\phi} = 1$.

        Attributes
        ----------
        metric: Callable[[torch.Tensor], torch.Tensor]
            The metric compression metric.
        name: str, default = 'prod_compression'
            Name of the prod compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the prod compression function.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method to define the compression function.

    """
    def __init__(self, name: str = 'prod_compression', *args, **kwargs):
        """
            The initialization method of the prod metric based compression function.

            It initializes the compression function based on the provided prod metric.

            Parameters
            ----------
            name: str, default = 'prod_compression'
                Name of the compression function.

            Returns
            ----------
            transformation
                The prod metric based compression function.
        """
        super().__init__(name=name, metric=batch_prod, *args, **kwargs)


class median_compression(metric_compression):
    r"""
        The median metric based compression function.

        It performs the data compression based on provided median metric.

        ...

        Notes
        ----------
        Formally, given a data instance $\mathbf{x} \in R^m$ and a provided metric $\phi: {R}^m \to {R}^{d_{\phi}}$,
        which transforms it into a dense representation of length $d_{\phi}$,
        we can represent the median metric based compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = median(\mathbf{x}) \in {R}^{d}.
            \end{equation}
        $$

        For the median metric studied in this project, the output is typically a scalar, i.e., the dimension $d = d_{\phi} = 1$.

        Attributes
        ----------
        metric: Callable[[torch.Tensor], torch.Tensor]
            The metric compression metric.
        name: str, default = 'median_compression'
            Name of the median compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the median compression function.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method to define the compression function.

    """
    def __init__(self, name: str = 'median_compression', *args, **kwargs):
        """
            The initialization method of the median metric based compression function.

            It initializes the compression function based on the provided median metric.

            Parameters
            ----------
            name: str, default = 'median_compression'
                Name of the compression function.

            Returns
            ----------
            transformation
                The median metric based compression function.
        """
        super().__init__(name=name, metric=batch_median, *args, **kwargs)
