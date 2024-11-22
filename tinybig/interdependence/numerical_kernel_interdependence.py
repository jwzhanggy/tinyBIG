# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################################
# Numerical kernel Interdependence #
####################################
"""
The numerical kernel based interdependence functions

This module contains the numerical kernel based interdependence functions, including
    numerical_kernel_based_interdependence,
    linear_kernel_interdependence,
    polynomial_kernel_interdependence,
    hyperbolic_tangent_kernel_interdependence,
    exponential_kernel_interdependence,
    minkowski_distance_interdependence,
    manhattan_distance_interdependence,
    euclidean_distance_interdependence,
    chebyshev_distance_interdependence,
    canberra_distance_interdependence,
    cosine_similarity_interdependence,
    gaussian_rbf_kernel_interdependence,
    laplacian_kernel_interdependence,
    anisotropic_rbf_kernel_interdependence,
    custom_hybrid_kernel_interdependence.
"""

import functools
from typing import Union, Any, Callable, List
import numpy as np

import torch

from tinybig.interdependence import interdependence

from tinybig.koala.linear_algebra import (
    linear_kernel,
    polynomial_kernel,
    hyperbolic_tangent_kernel,
    exponential_kernel,
    cosine_similarity_kernel,
    minkowski_distance_kernel,
    manhattan_distance_kernel,
    euclidean_distance_kernel,
    chebyshev_distance_kernel,
    canberra_distance_kernel,
    gaussian_rbf_kernel,
    laplacian_kernel,
    anisotropic_rbf_kernel,
    custom_hybrid_kernel,
)


class numerical_kernel_based_interdependence(interdependence):
    r"""
        A numerical kernel-based interdependence function.

        This class computes the interdependence matrix using a specified numerical kernel function.

        Notes
        ----------
        Formally, given a data batch $\mathbf{X} \in R^{b \times m}$, we define the numerical metric-based interdependence function as:
        $$
            \begin{equation}
            \xi(\mathbf{X}) = \mathbf{A} \in R^{m \times m'} \text{, where } \mathbf{A}(i, j) = \text{kernel} \left(\mathbf{X}(:, i), \mathbf{X}(:, j)\right).
            \end{equation}
        $$
        By convention, the resulting matrix $\mathbf{A}$ is square, with dimensions $m' = m$.


        Attributes
        ----------
        kernel : Callable
            The kernel function used to compute the interdependence matrix.

        Methods
        -------
        __init__(...)
            Initializes the kernel-based interdependence function.
        calculate_A(x=None, w=None, device='cpu', *args, **kwargs)
            Computes the interdependence matrix using the specified kernel.
    """

    def __init__(
        self,
        b: int, m: int, kernel: Callable,
        interdependence_type: str = 'attribute',
        name: str = 'kernel_based_interdependence',
        require_data: bool = True,
        require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        """
            Initializes the kernel-based interdependence function.

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
                Name of the interdependence function. Defaults to 'kernel_based_interdependence'.
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
            raise ValueError('the kernel is required for the kernel based interdependence function')
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
            ValueError
                If required data or parameters are missing.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            assert x is not None and x.ndim == 2
            x = self.pre_process(x=x, device=device)
            A = self.kernel(x)
            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                print(x.shape, A.shape, self.m, self.calculate_m_prime())
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A

            return A


class linear_kernel_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using a linear kernel.

        Notes
        ----------
        __Linear (Inner-Product) Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = \left\langle \mathbf{x}, \mathbf{y} \right \rangle.
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The linear kernel function.

        Methods
        -------
        __init__(...)
            Initializes the linear kernel interdependence function.
    """
    def __init__(self, name: str = 'linear_kernel_interdependence', *args, **kwargs):
        """
            Initializes the linear kernel interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'linear_kernel_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=linear_kernel, name=name, *args, **kwargs)


class cosine_similarity_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using a cosine similarity kernel.

        Notes
        ----------
        __Cosine Similarity based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = \frac{\left\langle \mathbf{x}, \mathbf{y} \right \rangle}{\left\|\mathbf{x}\right\| \cdot \left\| \mathbf{y} \right\|}.
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The cosine similarity kernel function.

        Methods
        -------
        __init__(...)
            Initializes the cosine similarity interdependence function.
    """
    def __init__(self, name: str = 'cosine_similarity_interdependence', *args, **kwargs):
        """
            Initializes the cosine similarity interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'cosine_similarity_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=cosine_similarity_kernel, name=name, *args, **kwargs)


class minkowski_distance_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using a Minkowski distance kernel.

        Notes
        ----------
        __Minkowski Distance based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = 1- \left\| \mathbf{x} - \mathbf{y} \right\|_p.
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The Minkowski distance kernel function.

        Methods
        -------
        __init__(...)
            Initializes the Minkowski distance interdependence function.
    """
    def __init__(self, p: Union[int, float, str, Any], name: str = 'minkowski_distance_interdependence', *args, **kwargs):
        """
            Initializes the Minkowski distance interdependence function.

            Parameters
            ----------
            p : Union[int, float, str, Any]
                The Minkowski distance parameter.
            name : str, optional
                Name of the interdependence function. Defaults to 'minkowski_distance_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        minkowski_kernel_func = functools.partial(minkowski_distance_kernel, p=p)
        super().__init__(kernel=minkowski_kernel_func, name=name, *args, **kwargs)


class manhattan_distance_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the Manhattan distance kernel.

        Notes
        ----------
        __Manhattan Distance based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = 1- \left\| \mathbf{x} - \mathbf{y} \right\|_1.
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The Manhattan distance kernel function.

        Methods
        -------
        __init__(...)
            Initializes the Manhattan distance interdependence function.
    """
    def __init__(self, name: str = 'manhattan_distance_interdependence', *args, **kwargs):
        """
            Initializes the Manhattan distance interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'manhattan_distance_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=manhattan_distance_kernel, name=name, *args, **kwargs)


class euclidean_distance_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the Euclidean distance kernel.

        Notes
        ----------
        __Euclidean Distance based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = 1- \left\| \mathbf{x} - \mathbf{y} \right\|_2.
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The Euclidean distance kernel function.

        Methods
        -------
        __init__(...)
            Initializes the Euclidean distance interdependence function.
    """
    def __init__(self, name: str = 'euclidean_distance_interdependence', *args, **kwargs):
        """
            Initializes the Euclidean distance interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'euclidean_distance_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=euclidean_distance_kernel, name=name, *args, **kwargs)


class chebyshev_distance_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the Chebyshev distance kernel.

        Notes
        ----------
        __Chebyshev Distance based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = 1- \left\| \mathbf{x} - \mathbf{y} \right\|_{\infty}.
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The Chebyshev distance kernel function.

        Methods
        -------
        __init__(...)
            Initializes the Chebyshev distance interdependence function.
    """
    def __init__(self, name: str = 'chebyshev_distance_interdependence', *args, **kwargs):
        """
            Initializes the Chebyshev distance interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'chebyshev_distance_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=chebyshev_distance_kernel, name=name, *args, **kwargs)


class canberra_distance_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the Canberra distance kernel.

        Notes
        ----------
        __Canberra Distance based Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = \sum_{i} \frac{|\mathbf{x}(i) - \mathbf{y}(i)|}{|\mathbf{x}(i)| + |\mathbf{y}(i)|}.
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The Canberra distance kernel function.

        Methods
        -------
        __init__(...)
            Initializes the Canberra distance interdependence function.
    """
    def __init__(self, name: str = 'canberra_distance_interdependence', *args, **kwargs):
        """
            Initializes the Canberra distance interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'canberra_distance_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(kernel=canberra_distance_kernel, name=name, *args, **kwargs)


class polynomial_kernel_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the polynomial kernel.

        Notes
        ----------
        __Polynomial Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y} | c, d) = \left(\left\langle \mathbf{x}, \mathbf{y} \right \rangle +c \right)^d.
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The polynomial kernel function.

        Methods
        -------
        __init__(...)
            Initializes the polynomial kernel interdependence function.
    """
    def __init__(self, name: str = 'polynomial_kernel_interdependence', c: float = 0.0, d: int = 1, *args, **kwargs):
        """
            Initializes the polynomial kernel interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'polynomial_kernel_interdependence'.
            c : float, optional
                Coefficient of the polynomial kernel. Defaults to 0.0.
            d : int, optional
                Degree of the polynomial kernel. Defaults to 1.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        polynomial_kernel_func = functools.partial(polynomial_kernel, c=c, d=d)
        super().__init__(kernel=polynomial_kernel_func, name=name, *args, **kwargs)


class hyperbolic_tangent_kernel_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the hyperbolic tangent kernel.

        Notes
        ----------
        __Hyperbolic Tangent Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y} | \alpha, c) = \text{tanh} \left( \alpha \left\langle \mathbf{x}, \mathbf{y} \right \rangle + c \right).
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The hyperbolic tangent kernel function.

        Methods
        -------
        __init__(...)
            Initializes the hyperbolic tangent kernel interdependence function.
    """
    def __init__(self, name: str = 'hyperbolic_tangent_kernel_interdependence', c: float = 0.0, alpha: float = 1.0, *args, **kwargs):
        """
            Initializes the hyperbolic tangent kernel interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'hyperbolic_tangent_kernel_interdependence'.
            c : float, optional
                Bias term of the hyperbolic tangent kernel. Defaults to 0.0.
            alpha : float, optional
                Scale factor of the hyperbolic tangent kernel. Defaults to 1.0.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        hyperbolic_tangent_kernel_func = functools.partial(hyperbolic_tangent_kernel, c=c, alpha=alpha)
        super().__init__(kernel=hyperbolic_tangent_kernel_func, name=name, *args, **kwargs)


class exponential_kernel_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the exponential kernel.

        Notes
        ----------
        __Exponential Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y} | \gamma) = \exp \left(- \gamma \left\| \mathbf{x} - \mathbf{y} \right\|_1 \right).
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The exponential kernel function.

        Methods
        -------
        __init__(...)
            Initializes the exponential kernel interdependence function.
    """
    def __init__(self, name: str = 'exponential_kernel_interdependence', gamma: float = 1.0, *args, **kwargs):
        """
            Initializes the exponential kernel interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'exponential_kernel_interdependence'.
            gamma : float, optional
                Scale factor of the exponential kernel. Defaults to 1.0.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        exponential_kernel_func = functools.partial(exponential_kernel, gamma=gamma)
        super().__init__(kernel=exponential_kernel_func, name=name, *args, **kwargs)


class gaussian_rbf_kernel_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the Gaussian RBF (Radial Basis Function) kernel.

        Notes
        ----------
        __Gaussian RBF Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y} | \sigma) = \exp \left(- \frac{\left\| \mathbf{x} - \mathbf{y} \right\|^2_2}{2 \sigma^2} \right).
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The Gaussian RBF kernel function.

        Methods
        -------
        __init__(...)
            Initializes the Gaussian RBF kernel interdependence function.
    """
    def __init__(self, name: str = 'gaussian_rbf_kernel_interdependence', sigma: float = 1.0, *args, **kwargs):
        """
            Initializes the Gaussian RBF kernel interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'gaussian_rbf_kernel_interdependence'.
            sigma : float, optional
                Standard deviation of the Gaussian RBF kernel. Defaults to 1.0.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        gaussian_rbf_kernel_func = functools.partial(gaussian_rbf_kernel, sigma=sigma)
        super().__init__(kernel=gaussian_rbf_kernel_func, name=name, *args, **kwargs)


class laplacian_kernel_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the Laplacian kernel.

        Notes
        ----------
        __Laplacian Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y} | \sigma) = \exp \left(- \frac{\left\| \mathbf{x} - \mathbf{y} \right\|_1}{ \sigma} \right).
            \end{equation}
        $$

        Attributes
        ----------
        kernel : Callable
            The Laplacian kernel function.

        Methods
        -------
        __init__(...)
            Initializes the Laplacian kernel interdependence function.
    """
    def __init__(self, name: str = 'laplacian_kernel_interdependence', sigma: float = 1.0, *args, **kwargs):
        """
            Initializes the Laplacian kernel interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'laplacian_kernel_interdependence'.
            sigma : float, optional
                Scale parameter for the Laplacian kernel. Defaults to 1.0.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        laplacian_kernel_func = functools.partial(laplacian_kernel, sigma=sigma)
        super().__init__(kernel=laplacian_kernel_func, name=name, *args, **kwargs)


class anisotropic_rbf_kernel_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using the anisotropic RBF (Radial Basis Function) kernel.

        Notes
        ----------
        __Anisotropic RBF Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y}) = \exp \left( - (\mathbf{x} - \mathbf{y}) \mathbf{A} (\mathbf{x} - \mathbf{y})^\top \right),
            \end{equation}
        $$
        where $\mathbf{A} = \text{diag}(\mathbf{a})$ is a diagonal matrix.

        Attributes
        ----------
        kernel : Callable
            The anisotropic RBF kernel function.

        Methods
        -------
        __init__(...)
            Initializes the anisotropic RBF kernel interdependence function.
    """
    def __init__(self, name: str = 'anisotropic_rbf_kernel_interdependence', a_vector: Union[torch.Tensor, np.array] = None, a_scalar: float = 1.0, *args, **kwargs):
        """
            Initializes the anisotropic RBF kernel interdependence function.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'anisotropic_rbf_kernel_interdependence'.
            a_vector : Union[torch.Tensor, np.array], optional
                Vector of scaling factors for each dimension. Defaults to None.
            a_scalar : float, optional
                Scalar scaling factor applied uniformly to all dimensions. Defaults to 1.0.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        anisotropic_rbf_kernel_func = functools.partial(anisotropic_rbf_kernel, a_vector=a_vector, a_scalar=a_scalar)
        super().__init__(kernel=anisotropic_rbf_kernel_func, name=name, *args, **kwargs)


class custom_hybrid_kernel_interdependence(numerical_kernel_based_interdependence):
    r"""
        A kernel-based interdependence class using a custom hybrid kernel.

        This class combines multiple kernel functions into a single hybrid kernel.

        Notes
        ----------
        __Custom Hybrid Kernel:__

        $$
            \begin{equation}
            \text{kernel}(\mathbf{x}, \mathbf{y} | \alpha, \beta) = \alpha k_1(\mathbf{x}, \mathbf{y}) + \beta k_2(\mathbf{x}, \mathbf{y}),
            \end{equation}
        $$
        where $k_1$ and $k_2$ are custom designed kernel, and $\alpha, \beta$ are the weights.

        Attributes
        ----------
        kernel : Callable
            The hybrid kernel function combining multiple kernels.
        weights : List[float], optional
            Weights applied to each kernel function.

        Methods
        -------
        __init__(...)
            Initializes the custom hybrid kernel interdependence function.
    """
    def __init__(self, kernels: List[Callable[[np.matrix], np.matrix]], weights: List[float] = None, name: str = 'custom_hybrid_kernel_interdependence', *args, **kwargs):
        """
            Initializes the custom hybrid kernel interdependence function.

            Parameters
            ----------
            kernels : List[Callable[[np.matrix], np.matrix]]
                List of kernel functions to combine.
            weights : List[float], optional
                Weights applied to each kernel function. Defaults to None, indicating equal weights.
            name : str, optional
                Name of the interdependence function. Defaults to 'custom_hybrid_kernel_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        custom_hybrid_kernel_func = functools.partial(custom_hybrid_kernel, kernels=kernels, weights=weights)
        super().__init__(kernel=custom_hybrid_kernel_func, name=name, *args, **kwargs)

