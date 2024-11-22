# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############################################
# Numerical Operator based Fusion Functions #
#############################################

"""
The metric based fusion functions

This module contains the metric based fusion function, such as
    metric_fusion,
    mean_fusion,
    prod_fusion,
    max_fusion,
    min_fusion,
    median_fusion,
    sum_fusion,
"""

from typing import Callable
import torch

from tinybig.fusion import fusion as base_fusion
from tinybig.koala.statistics import batch_mean, batch_weighted_mean, batch_std, batch_mode, batch_median, batch_entropy, batch_variance, batch_skewness, batch_harmonic_mean, batch_geometric_mean
from tinybig.koala.linear_algebra import batch_sum, batch_norm, batch_prod, batch_min, batch_max, batch_l1_norm, batch_l2_norm


class metric_fusion(base_fusion):
    r"""
        A fusion mechanism that applies a specified numerical/statistica metric across input tensors.

        Notes
        ----------

        Formally, given the input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of identical shapes,
        we can represent their fusion output as

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \mathbf{A}  \in R^{m \times n},
            \end{equation}
        $$

        where the entry $\mathbf{A}(i, j)$ (for $i \in \{1, 2, \cdots, m\}$ and $j \in \{1, 2, \cdots, n\}$) can be represented as

        $$
            \begin{equation}
            \mathbf{A}(i, j) = metric \left( \mathbf{A}_1(i,j), \mathbf{A}_2(i,j), \cdots, \mathbf{A}_k(i,j)  \right).
            \end{equation}
        $$

        The $metric(\cdots)$ can be either the numerical or statistical metrics, such as maximum, mean, product, etc.

        Attributes
        ----------
        metric : Callable[[torch.Tensor], torch.Tensor]
            A callable metric function to apply to the input tensors.

        Methods
        -------
        calculate_n(dims=None, *args, **kwargs)
            Computes the output dimension of the fused input.
        calculate_l(*args, **kwargs)
            Computes the number of learnable parameters, if applicable.
        forward(x, w=None, device='cpu', *args, **kwargs)
            Performs the metric-based fusion on the input tensors.
    """

    def __init__(
        self,
        dims: list[int] | tuple[int],
        metric: Callable[[torch.Tensor], torch.Tensor],
        name: str = "metric_fusion",
        *args, **kwargs
    ):
        """
            Initializes the metric-based fusion function.

            Parameters
            ----------
            dims : list[int] | tuple[int]
                Dimensions of the input tensors.
            metric : Callable[[torch.Tensor], torch.Tensor]
                A callable metric function to apply to the input tensors.
            name : str, optional
                Name of the fusion function. Defaults to "metric_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(dims=dims, name=name, require_parameters=False, *args, **kwargs)
        self.metric = metric

    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        """
            Computes the output dimension of the fused input.

            Parameters
            ----------
            dims : list[int] | tuple[int], optional
                List of dimensions of the input tensors. Defaults to None.

            Returns
            -------
            int
                Output dimension, equal to the input dimension if consistent.

            Raises
            ------
            AssertionError
                If input dimensions are inconsistent.
        """
        dims = dims if dims is not None else self.dims
        assert dims is not None and all(dim == dims[0] for dim in dims)
        return dims[0]

    def calculate_l(self, *args, **kwargs):
        """
            Computes the number of learnable parameters, if applicable.

            Returns
            -------
            int
                Number of learnable parameters. Returns 0 as metrics are non-parameterized.
        """
        return 0

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor], w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Performs the metric-based fusion on the input tensors.

            Parameters
            ----------
            x : list[torch.Tensor] | tuple[torch.Tensor]
                List or tuple of input tensors to be fused.
            w : torch.nn.Parameter, optional
                Learnable weights for fusion. Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Fused tensor after applying the metric.

            Raises
            ------
            ValueError
                If `x` is empty or if input tensors have inconsistent shapes.
            AssertionError
                If the metric is not callable.
        """
        if not x:
            raise ValueError("The input x cannot be empty...")
        if not all(x[0].shape == t.shape for t in x):
            raise ValueError("The input x must have the same shape.")

        x = torch.stack(x, dim=0)
        x = self.pre_process(x=x, device=device)

        assert self.metric is not None and isinstance(self.metric, Callable)

        x_shape = x.shape
        x_permuted = x.permute(*range(1, x.ndim), 0)

        fused_x = self.metric(x_permuted.view(-1, x_shape[0])).view(x_shape[1:])

        assert fused_x.size(-1) == self.calculate_n([element.size(-1) for element in x])
        return self.post_process(x=fused_x, device=device)


class max_fusion(metric_fusion):
    r"""
        A fusion mechanism that computes the element-wise maximum across input tensors.

        Notes
        ----------

        Formally, given the input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of identical shapes,
        we can represent their fusion output as

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \mathbf{A}  \in R^{m \times n},
            \end{equation}
        $$

        where the entry $\mathbf{A}(i, j)$ (for $i \in \{1, 2, \cdots, m\}$ and $j \in \{1, 2, \cdots, n\}$) can be represented as

        $$
            \begin{equation}
            \mathbf{A}(i, j) = max \left( \mathbf{A}_1(i,j), \mathbf{A}_2(i,j), \cdots, \mathbf{A}_k(i,j)  \right).
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the max fusion function.
    """
    def __init__(self, name: str = "max_fusion", *args, **kwargs):
        """
            Initializes the max fusion function.

            Parameters
            ----------
            name : str, optional
                Name of the fusion function. Defaults to "max_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class min_fusion(metric_fusion):
    r"""
        A fusion mechanism that computes the element-wise minimum across input tensors.

        Notes
        ----------

        Formally, given the input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of identical shapes,
        we can represent their fusion output as

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \mathbf{A}  \in R^{m \times n},
            \end{equation}
        $$

        where the entry $\mathbf{A}(i, j)$ (for $i \in \{1, 2, \cdots, m\}$ and $j \in \{1, 2, \cdots, n\}$) can be represented as

        $$
            \begin{equation}
            \mathbf{A}(i, j) = min \left( \mathbf{A}_1(i,j), \mathbf{A}_2(i,j), \cdots, \mathbf{A}_k(i,j)  \right).
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the min fusion function.
    """
    def __init__(self, name: str = "min_fusion", *args, **kwargs):
        """
            Initializes the min fusion function.

            Parameters
            ----------
            name : str, optional
                Name of the fusion function. Defaults to "min_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class sum_fusion(metric_fusion):
    r"""
        A fusion mechanism that computes the element-wise sum across input tensors.

        Notes
        ----------

        Formally, given the input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of identical shapes,
        we can represent their fusion output as

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \mathbf{A}  \in R^{m \times n},
            \end{equation}
        $$

        where the entry $\mathbf{A}(i, j)$ (for $i \in \{1, 2, \cdots, m\}$ and $j \in \{1, 2, \cdots, n\}$) can be represented as

        $$
            \begin{equation}
            \mathbf{A}(i, j) = sum \left( \mathbf{A}_1(i,j), \mathbf{A}_2(i,j), \cdots, \mathbf{A}_k(i,j)  \right).
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the sum fusion function.
    """
    def __init__(self, name: str = "sum_fusion", *args, **kwargs):
        """
            Initializes the sum fusion function.

            Parameters
            ----------
            name : str, optional
                Name of the fusion function. Defaults to "sum_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, metric=batch_sum, *args, **kwargs)


class mean_fusion(metric_fusion):
    r"""
        A fusion mechanism that computes the element-wise mean across input tensors.

        Notes
        ----------

        Formally, given the input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of identical shapes,
        we can represent their fusion output as

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \mathbf{A}  \in R^{m \times n},
            \end{equation}
        $$

        where the entry $\mathbf{A}(i, j)$ (for $i \in \{1, 2, \cdots, m\}$ and $j \in \{1, 2, \cdots, n\}$) can be represented as

        $$
            \begin{equation}
            \mathbf{A}(i, j) = mean \left( \mathbf{A}_1(i,j), \mathbf{A}_2(i,j), \cdots, \mathbf{A}_k(i,j)  \right).
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the mean fusion function.
    """
    def __init__(self, name: str = "mean_fusion", *args, **kwargs):
        """
            Initializes the mean fusion function.

            Parameters
            ----------
            name : str, optional
                Name of the fusion function. Defaults to "mean_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)


class prod_fusion(metric_fusion):
    r"""
        A fusion mechanism that computes the element-wise product across input tensors.

        Notes
        ----------

        Formally, given the input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of identical shapes,
        we can represent their fusion output as

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \mathbf{A}  \in R^{m \times n},
            \end{equation}
        $$

        where the entry $\mathbf{A}(i, j)$ (for $i \in \{1, 2, \cdots, m\}$ and $j \in \{1, 2, \cdots, n\}$) can be represented as

        $$
            \begin{equation}
            \mathbf{A}(i, j) = prod \left( \mathbf{A}_1(i,j), \mathbf{A}_2(i,j), \cdots, \mathbf{A}_k(i,j)  \right).
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the prod fusion function.
    """
    def __init__(self, name: str = "prod_fusion", *args, **kwargs):
        """
            Initializes the prod fusion function.

            Parameters
            ----------
            name : str, optional
                Name of the fusion function. Defaults to "prod_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, metric=batch_prod, *args, **kwargs)


class median_fusion(metric_fusion):
    r"""
        A fusion mechanism that computes the element-wise median across input tensors.

        Notes
        ----------

        Formally, given the input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of identical shapes,
        we can represent their fusion output as

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \mathbf{A}  \in R^{m \times n},
            \end{equation}
        $$

        where the entry $\mathbf{A}(i, j)$ (for $i \in \{1, 2, \cdots, m\}$ and $j \in \{1, 2, \cdots, n\}$) can be represented as

        $$
            \begin{equation}
            \mathbf{A}(i, j) = median \left( \mathbf{A}_1(i,j), \mathbf{A}_2(i,j), \cdots, \mathbf{A}_k(i,j)  \right).
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the median fusion function.
    """
    def __init__(self, name: str = "median_fusion", *args, **kwargs):
        """
            Initializes the median fusion function.

            Parameters
            ----------
            name : str, optional
                Name of the fusion function. Defaults to "median_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, metric=batch_median, *args, **kwargs)


class l1_norm_fusion(metric_fusion):
    r"""
        A fusion mechanism that computes the L1 norm across input tensors.

        Notes
        ----------

        Formally, given the input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of identical shapes,
        we can represent their fusion output as

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \mathbf{A}  \in R^{m \times n},
            \end{equation}
        $$

        where the entry $\mathbf{A}(i, j)$ (for $i \in \{1, 2, \cdots, m\}$ and $j \in \{1, 2, \cdots, n\}$) can be represented as

        $$
            \begin{equation}
            \mathbf{A}(i, j) = l1-norm \left( \mathbf{A}_1(i,j), \mathbf{A}_2(i,j), \cdots, \mathbf{A}_k(i,j)  \right).
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the L1 norm fusion function.
    """
    def __init__(self, name: str = "l1_norm_fusion", *args, **kwargs):
        """
            Initializes the L1 norm fusion function.

            Parameters
            ----------
            name : str, optional
                Name of the fusion function. Defaults to "l1_norm_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, metric=batch_l1_norm, *args, **kwargs)


class l2_norm_fusion(metric_fusion):
    r"""
        A fusion mechanism that computes the L2 norm across input tensors.

        Notes
        ----------

        Formally, given the input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of identical shapes,
        we can represent their fusion output as

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \mathbf{A}  \in R^{m \times n},
            \end{equation}
        $$

        where the entry $\mathbf{A}(i, j)$ (for $i \in \{1, 2, \cdots, m\}$ and $j \in \{1, 2, \cdots, n\}$) can be represented as

        $$
            \begin{equation}
            \mathbf{A}(i, j) = l2-norm \left( \mathbf{A}_1(i,j), \mathbf{A}_2(i,j), \cdots, \mathbf{A}_k(i,j)  \right).
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the L2 norm fusion function.
    """
    def __init__(self, name: str = "l2_norm_fusion", *args, **kwargs):
        """
            Initializes the L2 norm fusion function.

            Parameters
            ----------
            name : str, optional
                Name of the fusion function. Defaults to "l2_norm_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, metric=batch_l2_norm, *args, **kwargs)


