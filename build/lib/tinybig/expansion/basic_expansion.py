# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Basic data expansion functions.

This module contains the basic data expansion functions,
including identity_expansion, reciprocal_expansion and linear_expansion.
"""

import torch.nn

from tinybig.expansion import transformation

####################
# Basic Expansions #
####################


class identity_expansion(transformation):
    r"""
    The identity data expansion function.

    It performs the identity expansion of the input vector, and returns the expansion result.
    This class inherits from the expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For the identity expansion function, the expansion space dimension equals to the input space dimension.

    For input vector $\mathbf{x} \in R^m$, its identity expansion will be
    $$
        \begin{equation}
            \kappa(\mathbf{x}) = \sigma(\mathbf{x}) \in R^D
        \end{equation}
    $$
    where $D = m$. By default, we can also process the input with optional pre- or post-processing functions
    denoted by $\sigma(\cdot)$ in the above formula.

    Attributes
    ----------
    name: str, default = 'identity_expansion'
        Name of the expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='identity_expansion', *args, **kwargs):
        """
        The initialization method of the identity expansion function.

        It initializes an identity expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'identity_expansion'
            The name of the identity expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the identity expansion function, the expansion space dimension equals to the input space dimension, i.e.,
        $$ D = m. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs) -> torch.Tensor:
        r"""
        The forward method of the data expansion function.

        It performs the identity data expansion of the input data and returns the expansion result
        according to the following equation:
        $$
            \kappa(\mathbf{x}) = \mathbf{x} \in R^D,
        $$
        with optional pre- and post-processing functions.

        Examples
        ----------
        >>> import torch
        >>> from tinybig.expansion import identity_expansion
        >>> expansion = identity_expansion(name='identity_expansion')
        >>> x = torch.Tensor([0.5, 0.5])
        >>> kappa_x = expansion(x)
        >>> kappa_x
        tensor([0.5000, 0.5000])

        >>> import torch.nn.functional as F
        >>> expansion_with_preprocessing = identity_expansion(name='identity_expansion_with_preprocessing', preprocess_functions=F.relu)
        >>> kappa_x = expansion_with_preprocessing(x)
        >>> kappa_x
        tensor([0.5000, 0.5000])

        >>> expansion_with_postprocessing = identity_expansion(name='identity_expansion_with_postprocessing', postprocess_functions=F.sigmoid)
        >>> kappa_x = expansion_with_postprocessing(x)
        >>> kappa_x
        tensor([0.6225, 0.6225])

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.
        args: list, default = ()
            The other parameters.
        kwargs: dict, default = {}
            The other parameters.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        x = self.pre_process(x=x, device=device)
        expansion = x
        return self.post_process(x=expansion, device=device)


class reciprocal_expansion(transformation):
    r"""
    The reciprocal data expansion function.

    It performs the reciprocal expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its reciprocal expansion will be
    $$
        \begin{equation}
            \kappa(\mathbf{x}) = \frac{1}{\mathbf{x}} \in R^D
        \end{equation}
    $$
    where $D = m$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the reciprocal expansion function.

    Specifically, for very small positive and negative small values that are close to zero, the reciprocal
    expansion function will replace them with very small numbers $10^{-6}$ and $-10^{-6}$, respectively.
    In the current implementation, the input values in the range $[0, 10^{-6}]$ are replaced with $10^{-6}$,
    and values in the range $[-10^{-6}, 0)$ are replaced with $-10^{-6}$.

    Attributes
    ----------
    name: str, default = 'reciprocal_expansion'
        Name of the expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='reciprocal_expansion', *args, **kwargs):
        """
        The initialization method of the reciprocal expansion function.

        It initializes a reciprocal expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'reciprocal_expansion'
            The name of the reciprocal expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the reciprocal expansion function, the expansion space dimension equals to the input space dimension, i.e.,
        $$ D = m. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the reciprocal data expansion of the input data and returns the expansion result
        according to the following equation:
        $$
            \kappa(\mathbf{x}) = \frac{1}{\mathbf{x}} \in R^D
        $$
        with optional pre- and post-processing functions.


        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.
        args: list, default = ()
            The other parameters.
        kwargs: dict, default = {}
            The other parameters.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        x = self.pre_process(x=x, device=device)
        x[torch.logical_and(x>=0, x<=1e-6)] = 1e-6
        x[torch.logical_and(x<0, x>=-1e-6)] = -1e-6
        expansion = torch.reciprocal(x)
        return self.post_process(x=expansion, device=device)


class linear_expansion(transformation):
    r"""
    The linear data expansion function.

    It performs the linear expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its linear expansion can be based on one of the following equations:
    $$
    \begin{align}
        \kappa(\mathbf{x}) &= c \mathbf{x} \in {R}^D, \\\\
        \kappa(\mathbf{x}) &= \mathbf{x} \mathbf{C}\_{post} \in {R}^D,\\\\
        \kappa(\mathbf{x}) &= \mathbf{C}\_{pre} \mathbf{x} \in {R}^{D},
    \end{align}
    $$
    where $c \in {R}$, $\mathbf{C}_{post}, \mathbf{C}_{pre} \in {R}^{m \times m}$ denote the provided
    constant scalar and linear transformation matrices, respectively.
    Linear data expansion will not change the data vector dimensions, and the output data vector dimension $D=m$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the linear expansion function.

    Attributes
    ----------
    name: str, default = 'linear_expansion'
        Name of the expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='linear_expansion', c=None, pre_C=None, post_C=None, *args, **kwargs):
        r"""
        The initialization method of the linear expansion function.

        It initializes a linear expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'linear_expansion'
            The name of the linear expansion function.
        c: float | torch.Tensor, default = None
            The scalar $c$ of the linear expansion.
        pre_C: torch.Tensor, default = None
            The $\mathbf{C}_{pre}$ matrix of the linear expansion.
        post_C: torch.Tensor, default = None
            The $\mathbf{C}_{post}$ matrix of the linear expansion.
        """
        super().__init__(name=name, *args, **kwargs)
        self.c = c
        self.pre_C = pre_C
        self.post_C = post_C

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the linear expansion function, the expansion space dimension equals to the input space dimension, i.e.,
        $$ D = m. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m

    def forward(self, x: torch.Tensor, device='cpu', c=None, pre_C=None, post_C=None, *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the linear data expansion of the input data and returns the expansion result
        according to one of the following equation:
        $$
        \begin{align}
            \kappa(\mathbf{x}) &= c \mathbf{x} \in {R}^D, \\\\
            \kappa(\mathbf{x}) &= \mathbf{x} \mathbf{C}\_{post} \in {R}^D,\\\\
            \kappa(\mathbf{x}) &= \mathbf{C}\_{pre} \mathbf{x} \in {R}^{D},
        \end{align}
        $$
        where $c \in {R}$, $\mathbf{C}_{post}, \mathbf{C}_{pre} \in {R}^{m \times m}$ denote the provided
        constant scalar and linear transformation matrices, respectively.


        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.
        c: float | torch.Tensor, default = None
            The scalar $c$ of the linear expansion.
        pre_C: torch.Tensor, default = None
            The $\mathbf{C}_{pre}$ matrix of the linear expansion.
        post_C: torch.Tensor, default = None
            The $\mathbf{C}_{post}$ matrix of the linear expansion.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        x = self.pre_process(x=x, device=device)

        c = c if c is not None else self.c
        pre_C = pre_C if pre_C is not None else self.pre_C
        post_C = post_C if post_C is not None else self.post_C

        if c is not None:
            expansion = c * x
        elif pre_C is not None:
            assert pre_C.size(-1) == x.size(0)
            expansion = torch.matmul(pre_C, x)
        elif post_C is not None:
            assert x.size(-1) == post_C.size(0)
            expansion = torch.matmul(x, post_C)
        else:
            expansion = x

        return self.post_process(x=expansion, device=device)
