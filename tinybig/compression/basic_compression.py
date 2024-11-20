# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Basic Compression Functions #
###############################

"""
Basic data compression functions.

This module contains the basic data compression functions,
including identity_compression, reciprocal_compression and linear_compression.
"""

from tinybig.expansion import identity_expansion, reciprocal_expansion, linear_expansion


class identity_compression(identity_expansion):
    r"""
        The identity data compression function.

        It performs the identity compression of the input data batch, and returns the compression result.
        This class directly inherits from the identity_expansion class, which inherits from the base data transformation class.
        Both of them are performing the identical data transformation operators on the input data batch.

        ...

        Notes
        ----------
        For the identity compression function, the compression space dimension equals to the input space dimension.

        For input vector $\mathbf{x} \in R^m$, its identity compression will be
        $$
            \begin{equation}
                \kappa(\mathbf{x}) = \sigma(\mathbf{x}) \in R^d
            \end{equation}
        $$
        where $d = m$. By default, we can also process the input with optional pre- or post-processing functions
        denoted by $\sigma(\cdot)$ in the above formula.

        Attributes
        ----------
        name: str, default = 'identity_compression'
            Name of the compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the compression function.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method declared in the base transformation class.

    """

    def __init__(self, name='identity_compression', *args, **kwargs):
        """
            The initialization method of the identity compression function.

            It initializes an identity compression object based on the input function name.
            This method will also call the initialization method of the base class.

            Parameters
            ----------
            name: str, default = 'identity_compression'
                The name of the identity compression function.

            Returns
            ----------
            transformation
                The identity compression function.
        """
        super().__init__(name=name, *args, **kwargs)


class reciprocal_compression(reciprocal_expansion):
    r"""
        The reciprocal data compression function.

        It performs the reciprocal compression of the input vector, and returns the compression result.
        The class inherits from the reciprocal_expansion class, which inherits from the base data transformation class.

        ...

        Notes
        ----------
        For input vector $\mathbf{x} \in R^m$, its reciprocal compression will be
        $$
            \begin{equation}
                \kappa(\mathbf{x}) = \frac{1}{\mathbf{x}} \in R^d
            \end{equation}
        $$
        where $d = m$.

        By default, the input and output can also be processed with the optional pre- or post-processing functions
        in the reciprocal compression function.

        Specifically, for very small positive and negative small values that are close to zero, the reciprocal
        compression function will replace them with very small numbers $10^{-6}$ and $-10^{-6}$, respectively.
        In the current implementation, the input values in the range $[0, 10^{-6}]$ are replaced with $10^{-6}$,
        and values in the range $[-10^{-6}, 0)$ are replaced with $-10^{-6}$.

        Attributes
        ----------
        name: str, default = 'reciprocal_compression'
            Name of the compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the compression function.

        calculate_D
            It calculates the compression space dimension d based on the input dimension parameter m.

        forward
            It implements the abstract forward method declared in the base compression class.

    """
    def __init__(self, name='reciprocal_compression', *args, **kwargs):
        """
            The initialization method of the reciprocal compression function.

            It initializes a reciprocal compression object based on the input function name.
            This method will also call the initialization method of the base class as well.

            Parameters
            ----------
            name: str, default = 'reciprocal_compression'
                The name of the reciprocal compression function.

            Returns
            ----------
            transformation
                The reciprocal compression function.
        """
        super().__init__(name=name, *args, **kwargs)


class linear_compression(linear_expansion):
    r"""
        The linear data compression function.

        It performs the linear compression of the input vector, and returns the compression result.
        The class inherits from the linear_expansion class, which inherits from the base data transformation class.

        ...

        Notes
        ----------
        For input vector $\mathbf{x} \in R^m$, its linear compression can be based on one of the following equations:
        $$
        \begin{align}
            \kappa(\mathbf{x}) &= c \mathbf{x} \in {R}^d, \\\\
            \kappa(\mathbf{x}) &= \mathbf{x} \mathbf{C}\_{post} \in {R}^d,\\\\
            \kappa(\mathbf{x}) &= \mathbf{C}\_{pre} \mathbf{x} \in {R}^{d},
        \end{align}
        $$
        where $c \in {R}$, $\mathbf{C}_{post}, \mathbf{C}_{pre} \in {R}^{m \times m}$ denote the provided
        constant scalar and linear transformation matrices, respectively.
        Linear data compression will not change the data vector dimensions, and the output data vector dimension $d=m$.

        By default, the input and output can also be processed with the optional pre- or post-processing functions
        in the linear compression function.

        Attributes
        ----------
        name: str, default = 'linear_compression'
            Name of the compression function.

        Methods
        ----------
        __init__
            It performs the initialization of the compression function.

        calculate_D
            It calculates the compression space dimension D based on the input dimension parameter m.

        forward
            It implements the abstract forward method declared in the base compression class.

    """
    def __init__(self, name='linear_compression', c=None, pre_C=None, post_C=None, *args, **kwargs):
        r"""
            The initialization method of the linear compression function.

            It initializes a linear compression object based on the input function name.
            This method will also call the initialization method of the base class as well.

            Parameters
            ----------
            name: str, default = 'linear_compression'
                The name of the linear compression function.
            c: float | torch.Tensor, default = None
                The scalar $c$ of the linear compression.
            pre_C: torch.Tensor, default = None
                The $\mathbf{C}_{pre}$ matrix of the linear compression.
            post_C: torch.Tensor, default = None
                The $\mathbf{C}_{post}$ matrix of the linear compression.

            Returns
            ----------
            transformation
                The linear compression function.
        """
        super().__init__(name=name, c=c, pre_C=pre_C, post_C=post_C, *args, **kwargs)
