# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

######################################
# Statistical kernel Interdependence #
######################################
"""
The statistical kernel based interdependence functions

This module contains the statistical kernel based interdependence functions, including
    statistical_kernel_based_interdependence,
    kl_divergence_interdependence,
    pearson_correlation_interdependence,
    rv_coefficient_interdependence,
    mutual_information_interdependence.
"""

import torch
from typing import Callable

from tinybig.interdependence import interdependence
from tinybig.koala.statistics import (
    batch_kl_divergence_kernel,
    batch_pearson_correlation_kernel,
    batch_rv_coefficient_kernel,
    batch_mutual_information_kernel
)


class statistical_kernel_based_interdependence(interdependence):
    r"""
        A statistical kernel-based interdependence function.

        This class computes the interdependence matrix using a specified statistical kernel function.

        Notes
        ----------

        Formally, given a data batch $\mathbf{X} \in R^{b \times m}$, we can define the statistical kernel-based interdependence function as:

        $$
            \begin{equation}
            \xi(\mathbf{X}) = \mathbf{A} \in R^{m \times m'} \text{, where } \mathbf{A}(i, j) = \text{kernel} \left(\mathbf{X}(:, i), \mathbf{X}(:, j)\right).
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The kernel function used to compute the interdependence matrix.

        Methods
        -------
        __init__(...)
            Initializes the statistical kernel-based interdependence function.
        calculate_A(x=None, w=None, device='cpu', *args, **kwargs)
            Computes the interdependence matrix using the specified kernel.
    """

    def __init__(
        self,
        b: int, m: int, kernel: Callable,
        interdependence_type: str = 'attribute',
        name: str = 'statistical_kernel_based_interdependence',
        require_data: bool = True,
        require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        """
            Initializes the statistical kernel-based interdependence function.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            kernel : Callable
                The kernel function used to compute the interdependence matrix.
            interdependence_type : str, optional
                Type of interdependence ('attribute', 'instance', etc.). Defaults to 'attribute'.
            name : str, optional
                Name of the interdependence function. Defaults to 'statistical_kernel_based_interdependence'.
            require_data : bool, optional
                If True, requires input data for matrix computation. Defaults to True.
            require_parameters : bool, optional
                If True, requires parameters for matrix computation. Defaults to False.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            ValueError
                If no kernel function is provided.
        """
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_parameters=require_parameters, require_data=require_data, device=device, *args, **kwargs)

        if kernel is None:
            raise ValueError('the kernel is required for the statistical kernel based interdependence function')
        self.kernel = kernel
        self.kernel = kernel

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the interdependence matrix using the specified kernel function.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Required for computation. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor. Defaults to None.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for interdependence functions.
            **kwargs : dict
                Additional keyword arguments for interdependence functions.

            Returns
            -------
            torch.Tensor
                The computed interdependence matrix.

            Raises
            ------
            AssertionError
                If `x` is not provided or has an incorrect shape.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            assert x is not None and x.ndim == 2
            x = self.pre_process(x=x, device=device)
            A = self.kernel(x)
            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A

            return A


class kl_divergence_interdependence(statistical_kernel_based_interdependence):
    r"""
        A statistical kernel-based interdependence class using the KL divergence kernel.

        Notes
        ----------
        __KL Divergence based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = \sum_i \mathbf{x}(i) \log \left(\frac{\mathbf{x}(i)}{\mathbf{y}(i)} \right),
            \end{equation}
        $$
        where $\mb{x}$ and $\mb{y}$ have been normalized.

        Attributes
        ----------
        kernel : Callable
            The KL divergence kernel function.

        Methods
        -------
        __init__(...)
            Initializes the KL divergence interdependence function.
    """
    def __init__(self, name: str = 'kl_divergence_interdependence', *args, **kwargs):
        """
            Initializes the KL divergence interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'kl_divergence_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=batch_kl_divergence_kernel, name=name, *args, **kwargs)


class pearson_correlation_interdependence(statistical_kernel_based_interdependence):
    r"""
        A statistical kernel-based interdependence class using the Pearson correlation kernel.

        Notes
        ----------
        __Pearson Correlation based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = \frac{\sum_{i=1}^b (\frac{\mathbf{x}(i) - \mu_x}{\sigma_x} ) (\frac{\mathbf{y}(i) - \mu_y}{\sigma_y} )}{b},
            \end{equation}
        $$
        where $\mu_x, \mu_y, \sigma_x, \sigma_y$ are the mean and std.

        Attributes
        ----------
        kernel : Callable
            The Pearson correlation kernel function.

        Methods
        -------
        __init__(...)
            Initializes the Pearson correlation interdependence function.
    """
    def __init__(self, name: str = 'pearson_correlation_interdependence', *args, **kwargs):
        """
            Initializes the Pearson correlation interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'pearson_correlation_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=batch_pearson_correlation_kernel, name=name, *args, **kwargs)


class rv_coefficient_interdependence(statistical_kernel_based_interdependence):
    r"""
        A statistical kernel-based interdependence class using the RV coefficient kernel.

        Notes
        ----------
        __RV coefficient based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = \frac{ tr(\Sigma_{x,y} \Sigma_{y,x} ) }{ \sqrt{ tr(\Sigma_x^2) tr(\Sigma_y^2)}}.
            \end{equation}
        $$
        where $\Sigma_{x}$, $\Sigma_{y}$ and $\Sigma_{x,y}$ denote the variance/co-variance matrix of $\mathbf{x}$ and $\mathbf{y}$ respectively.

        Attributes
        ----------
        kernel : Callable
            The RV coefficient kernel function.

        Methods
        -------
        __init__(...)
            Initializes the RV coefficient interdependence function.
    """
    def __init__(self, name: str = 'rv_coefficient_interdependence', *args, **kwargs):
        """
            Initializes the RV coefficient interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'rv_coefficient_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=batch_rv_coefficient_kernel, name=name, *args, **kwargs)


class mutual_information_interdependence(statistical_kernel_based_interdependence):
    r"""
        A statistical kernel-based interdependence class using the mutual information kernel.

        Notes
        ----------
        __Mutual Information based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = \frac{1}{2} \log \left( \frac{ det(\Sigma_x) det( \Sigma_y) }{det \left( \Sigma \right)} \right).
            \end{equation}
        $$

        where $\Sigma_{x}$ and $\Sigma_{y}$ denote the variance matrix of $\mathbf{x}$ and $\mathbf{y}$ respectively.

        Notation $\Sigma$ is the co-variance matrix of the joint variables $\mathbf{x}$ and $\mathbf{y}$, which can be represented as follows:

        $$
        \begin{equation}
            \Sigma = \begin{bmatrix} \Sigma_x & \Sigma_{x,y} \\ \Sigma_{y,x} & \Sigma_y \end{bmatrix}.
        \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The mutual information kernel function.

        Methods
        -------
        __init__(...)
            Initializes the mutual information interdependence function.
    """
    def __init__(self, name: str = 'mutual_information_interdependence', *args, **kwargs):
        """
            Initializes the mutual information interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'mutual_information_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=batch_mutual_information_kernel, name=name, *args, **kwargs)


