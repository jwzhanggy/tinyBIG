# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#########################
# Basic Interdependence #
#########################

"""
The basic interdependence functions

This module contains the basic interdependence functions, including
    constant_interdependence,
    constant_c_interdependence,
    zero_interdependence,
    one_interdependence,
    identity_interdependence.
"""

import warnings
import torch

from tinybig.interdependence import interdependence


class constant_interdependence(interdependence):
    r"""
        A class for constant interdependence.

        This class defines a constant interdependence matrix (`A`) for the relationship between rows or columns
        of the input tensor. It does not require input data or additional parameters for computation.

        Notes
        ----------
        Formally, based on the (optional) input data batch $\mathbf{X} \in {R}^{b \times m}$, we define the constant interdependence function as:

        \begin{equation}
        \xi(\mathbf{X}) = \mathbf{A} \in {R}^{m \times m'}.
        \end{equation}

        This function facilitates the definition of customized constant interdependence matrices, allowing for a
        manually defined matrix $\mathbf{A}$ to be provided as a hyper-parameter during function initialization.

        Two special cases warrant particular attention: when $\mathbf{A}_c$ consists entirely of zeros, it is designated as
        the "zero interdependence matrix", whereas a matrix of all ones is termed the "one interdependence matrix".

        Attributes
        ----------
        A : torch.Tensor
            The interdependence matrix of shape `(b, b_prime)` or `(m, m_prime)`, depending on the interdependence type.
        b : int
            Number of rows in the input tensor.
        m : int
            Number of columns in the input tensor.
        interdependence_type : str
            Type of interdependence ('attribute', 'instance', etc.).
        name : str
            Name of the interdependence function.
        device : str
            Device for computation (e.g., 'cpu' or 'cuda').

        Methods
        -------
        __init__(b, m, A, interdependence_type='attribute', name='constant_interdependence', ...)
            Initializes the constant interdependence function.
        update_A(A)
            Updates the interdependence matrix `A`.
        calculate_b_prime(b=None)
            Computes the number of rows in the output tensor after interdependence.
        calculate_m_prime(m=None)
            Computes the number of columns in the output tensor after interdependence.
        calculate_A(x=None, w=None, device='cpu', ...)
            Returns the constant interdependence matrix `A`.
    """
    def __init__(
        self,
        b: int, m: int,
        A: torch.Tensor,
        interdependence_type: str = 'attribute',
        name: str = 'constant_interdependence',
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
            Initializes the constant interdependence function.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            A : torch.Tensor
                The interdependence matrix of shape `(b, b_prime)` or `(m, m_prime)`, depending on the interdependence type.
            interdependence_type : str, optional
                Type of interdependence ('attribute', 'instance', etc.). Defaults to 'attribute'.
            name : str, optional
                Name of the interdependence function. Defaults to 'constant_interdependence'.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for the parent `interdependence` class.
            **kwargs : dict
                Additional keyword arguments for the parent `interdependence` class.

            Raises
            ------
            ValueError
                If `A` is None or does not have 2 dimensions.
        """
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=False, require_parameters=False, device=device, *args, **kwargs)
        if A is None or A.ndim != 2:
            raise ValueError('The parameter matrix A is required and should have ndim: 2 by default')
        self.A = A
        if self.A.device != device:
            self.A.to(device)

    def update_A(self, A: torch.Tensor):
        """
            Updates the interdependence matrix `A`.

            Parameters
            ----------
            A : torch.Tensor
                The new interdependence matrix of shape `(b, b_prime)` or `(m, m_prime)`.

            Raises
            ------
            ValueError
                If `A` is None or does not have 2 dimensions.
        """
        if A is None or A.ndim != 2:
            raise ValueError('The parameter matrix A is required and should have ndim: 2 by default')
        self.check_A_shape_validity(A=A)
        self.A = A

    def calculate_b_prime(self, b: int = None):
        """
            Computes the number of rows in the output tensor after applying interdependence function.

            Parameters
            ----------
            b : int, optional
                Number of rows in the input tensor. If None, defaults to `self.b`.

            Returns
            -------
            int
                The number of rows in the output tensor.

            Raises
            ------
            AssertionError
                If `b` does not match the shape of `A` for row-based interdependence.
        """
        b = b if b is not None else self.b
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert self.A is not None and b is not None and self.A.size(0) == b
            return self.A.size(1)
        else:
            return b

    def calculate_m_prime(self, m: int = None):
        """
            Computes the number of columns in the output tensor after applying interdependence function.

            Parameters
            ----------
            m : int, optional
                Number of columns in the input tensor. If None, defaults to `self.m`.

            Returns
            -------
            int
                The number of columns in the output tensor.

            Raises
            ------
            AssertionError
                If `m` does not match the shape of `A` for column-based interdependence.
        """
        m = m if m is not None else self.m
        if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert self.A is not None and m is not None and self.A.size(0) == m
            return self.A.size(1)
        else:
            return m

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Returns the constant interdependence matrix `A`.

            Parameters
            ----------
            x : torch.Tensor, optional
                Ignored for constant interdependence. Defaults to None.
            w : torch.nn.Parameter, optional
                Ignored for constant interdependence. Defaults to None.
            device : str, optional
                Device for computation. Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The constant interdependence matrix `A`.

            Raises
            ------
            AssertionError
                If `A` is not set or requires data or parameters for computation.
        """
        assert self.A is not None and self.require_data is False and self.require_parameters is False
        return self.A


class constant_c_interdependence(constant_interdependence):
    r"""
        A class for constant interdependence with a scalar multiplier.

        This class defines an interdependence matrix as a scalar (`c`) multiplied by a matrix of ones.

        Notes
        -------
        Formally, based on the (optional) input data batch $\mathbf{X} \in {R}^{b \times m}$, we define the constant interdependence function as:
        $$
            \begin{equation}
            \xi(\mathbf{X}) = c \times \mathbf{1} \in {R}^{m \times m}, \text{ or } \xi(\mathbf{X}) = c \times \mathbf{1} \in {R}^{b \times b},
            \end{equation}
        $$
        where $c$ is the provided constant factor and $\mathbf{1}$ is a matrix of ones.

        Methods
        -------
        __init__(b, m, b_prime=None, m_prime=None, c=1.0, ...)
            Initializes the constant-c interdependence function.
    """

    def __init__(
        self,
        b: int, m: int,
        b_prime: int = None, m_prime: int = None,
        c: float | int = 1.0,
        name: str = 'constant_c_interdependence',
        interdependence_type: str = 'attribute',
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
            Initializes the constant-c interdependence function.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            b_prime : int, optional
                Number of rows in the output tensor for row-based interdependence. Required for row-based interdependence types.
            m_prime : int, optional
                Number of columns in the output tensor for column-based interdependence. Required for column-based interdependence types.
            c : float or int, optional
                Scalar multiplier for the interdependence matrix. Defaults to 1.0.
            name : str, optional
                Name of the interdependence function. Defaults to 'constant_c_interdependence'.
            interdependence_type : str, optional
                Type of interdependence ('attribute', 'instance', etc.). Defaults to 'attribute'.
            device : str, optional
                Device for computation. Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for the parent `constant_interdependence` class.
            **kwargs : dict
                Additional keyword arguments for the parent `constant_interdependence` class.

            Raises
            ------
            ValueError
                If the interdependence type is not supported.
        """

        if interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert b_prime is not None
            A = c * torch.ones((b, b_prime), device=device)
        elif interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert m_prime is not None
            A = c * torch.ones((m, m_prime), device=device)
        else:
            raise ValueError(f'Interdependence type {interdependence_type} is not supported')
        super().__init__(b=b, m=m, A=A, name=name, interdependence_type=interdependence_type, device=device, *args, **kwargs)


class zero_interdependence(constant_c_interdependence):
    r"""
        A class for zero interdependence.

        This class defines an interdependence matrix filled with zeros, which inherits from `constant_c_interdependence` class
        with constant factor $c$ set to zero.

        Notes
        -------
        Formally, based on the (optional) input data batch $\mathbf{X} \in {R}^{b \times m}$, we define the zero interdependence function as:
        $$
            \begin{equation}
            \xi(\mathbf{X}) = 0.0 \times \mathbf{1} \in {R}^{m \times m}, \text{ or } \xi(\mathbf{X}) = 0.0 \times \mathbf{1} \in {R}^{b \times b}.
            \end{equation}
        $$

        Methods
        -------
        __init__(name='zero_interdependence', ...)
            Initializes the zero interdependence function.
    """
    def __init__(self, name: str = 'zero_interdependence', *args, **kwargs):
        """
            Initializes the zero interdependence function.

            This class sets the interdependence matrix to be filled with zeros, ensuring no relationship
            between the rows or columns of the input tensor.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'zero_interdependence'.
            *args : tuple
                Additional positional arguments for the parent `constant_c_interdependence` class.
            **kwargs : dict
                Additional keyword arguments for the parent `constant_c_interdependence` class.
        """
        super().__init__(c=0.0, name=name, *args, **kwargs)


class one_interdependence(constant_c_interdependence):
    r"""
        A class for one interdependence.

        This class defines an interdependence matrix filled with zeros, which inherits from `constant_c_interdependence` class
        with constant factor $c$ set to one.

        Notes
        -------
        Formally, based on the (optional) input data batch $\mathbf{X} \in {R}^{b \times m}$, we define the one interdependence function as:
        $$
            \begin{equation}
            \xi(\mathbf{X}) = 1.0 \times \mathbf{1} \in {R}^{m \times m}, \text{ or } \xi(\mathbf{X}) = 1.0 \times \mathbf{1} \in {R}^{b \times b}.
            \end{equation}
        $$

        Methods
        -------
        __init__(name='one_interdependence', ...)
            Initializes the one interdependence function.
    """

    def __init__(self, name: str = 'one_interdependence', *args, **kwargs):
        """
            Initializes the one interdependence function.

            This class sets the interdependence matrix to be filled with ones, representing a uniform
            relationship between all rows or columns of the input tensor.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'one_interdependence'.
            *args : tuple
                Additional positional arguments for the parent `constant_c_interdependence` class.
            **kwargs : dict
                Additional keyword arguments for the parent `constant_c_interdependence` class.
        """
        super().__init__(c=1.0, name=name, *args, **kwargs)


class identity_interdependence(constant_interdependence):
    r"""
        A class for identity interdependence.

        This class defines an identity interdependence matrix, preserving the input dimensions.

        Notes
        -------
        Formally, based on the (optional) input data batch $\mathbf{X} \in {R}^{b \times m}$, we define the one interdependence function as:
        $$
            \begin{equation}
            \xi(\mathbf{X}) = \mathbf{I} \in {R}^{m \times m}, \text{ or } \xi(\mathbf{X}) = \mathbf{I} \in {R}^{b \times b},
            \end{equation}
        $$
        where $\mathbf{I}$ denotes the identity interdependence matrix.

        Methods
        -------
        __init__(b, m, b_prime=None, m_prime=None, ...)
            Initializes the identity interdependence function.
    """
    def __init__(
        self,
        b: int, m: int,
        b_prime: int = None, m_prime: int = None,
        name: str = 'identity_interdependence',
        interdependence_type: str = 'attribute',
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
            Initializes the identity interdependence function.

            This class sets the interdependence matrix to be an identity matrix, preserving the
            dimensions and structure of the input tensor.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            b_prime : int, optional
                Number of rows in the output tensor for row-based interdependence. Defaults to `b`.
            m_prime : int, optional
                Number of columns in the output tensor for column-based interdependence. Defaults to `m`.
            name : str, optional
                Name of the interdependence function. Defaults to 'identity_interdependence'.
            interdependence_type : str, optional
                Type of interdependence ('attribute', 'instance', etc.). Defaults to 'attribute'.
            device : str, optional
                Device for computation. Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for the parent `constant_interdependence` class.
            **kwargs : dict
                Additional keyword arguments for the parent `constant_interdependence` class.

            Raises
            ------
            ValueError
                If the interdependence type is not supported.
            Warning
                If `b != b_prime` or `m != m_prime`, indicating that the interdependence is not strictly identity.
        """
        b_prime = b_prime if b_prime is not None else b
        m_prime = m_prime if m_prime is not None else m

        if interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert b_prime is not None
            A = torch.eye(b, b_prime, device=device)
            if b != b_prime:
                warnings.warn("b and b_prime are different, this function will change the row dimensions of the inputs and cannot guarantee identity interdependence...")
        elif interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert m_prime is not None
            A = torch.eye(m, m_prime, device=device)
            if m != m_prime:
                warnings.warn("m and m_prime are different, this function will change the column dimensions of the inputs and cannot guarantee identity interdependence...")

        else:
            raise ValueError(f'Interdependence type {interdependence_type} is not supported')
        super().__init__(b=b, m=m, A=A, name=name, interdependence_type=interdependence_type, device=device, *args, **kwargs)

