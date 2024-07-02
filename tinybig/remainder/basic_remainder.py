# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Basic remainder functions.

This module contains the basic remainder functions, including constant_remainder, zero_remainder, one_remainder,
identity_remainder, and linear_remainder.
"""

import torch
import torch.nn.functional as F

from tinybig.remainder import remainder


####################
# Basic Remainders #
####################


class constant_remainder(remainder):
    r"""
    The constant remainder function.

    It calculates the constant remainder, and returns remainder term of certain length.
    This class inherits from the base remainder class (i.e., the remainder class in the module directory).

    ...

    Notes
    ----------
    The constant remainder function $\pi: {R}^m \to {R}^n$ just projects all inputs to a constant vector, i.e.,
    $$
        \begin{equation}
            \pi(\mathbf{x}) = \mathbf{c} = c \times \mathbf{1}^n \in {R}^n,
        \end{equation}
    $$
    where $\mathbf{c} = c \times \mathbf{1}^n$ is a constant vector of length $n$.

    By default, the remainder term will also be processed with the optional activation functions.

    Attributes
    ----------
    name: str, default = 'constant_remainder'
        Name of the remainder function.
    c: float, default = 1.0
        The constant value of the remainder function

    Methods
    ----------
    __init__
        It initializes the constant remainder function.

    forward
        It implements the abstract forward method declared in the base remainder class.
    """
    def __init__(self, name='constant_remainder', c=1.0, *args, **kwargs):
        """
        The initialization method of the constant remainder function.

        It initializes a constant remainder function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'constant_remainder'
            Name of the constant remainder function.
        c: float, default = 1.0
            The constant value of the remainder function

        Returns
        ----------
        object
            The constant remainder function object.
        """
        super().__init__(name=name, *args, **kwargs)
        self.c = c

    def forward(self, n: int, device='cpu', *args, **kwargs):
        r"""
        The forward method of the constant remainder function.

        It calculates the constant remainder term of length $n$, which can be represented as follows
        $$
            \begin{equation}
                \pi(\mathbf{x}) = \mathbf{c} \in {R}^n,
            \end{equation}
        $$

        By default, the remainder term will also be processed with the optional activation functions.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        device: str, default = 'cpu'
            Device to calculate the remainder function.

        Returns
        ----------
        torch.Tensor
            The remainder term of length $n$.
        """
        x = self.c * torch.ones(n)
        return self.activation(x=x, device=device)


class zero_remainder(constant_remainder):
    r"""
    The zero remainder function.

    As a special case of constant remainder, the zero remainder returns constant zero vector as the remainder term.
    This class inherits from the constant remainder class.

    ...

    Notes
    ----------
    The zero remainder function $\pi: {R}^m \to {R}^n$ just projects all inputs to a constant zero vector, i.e.,
    $$
        \begin{equation}
            \pi(\mathbf{x}) = \mathbf{0}^n \in {R}^n.
        \end{equation}
    $$

    By default, the remainder term will also be processed with the optional activation functions.

    Attributes
    ----------
    name: str, default = 'zero_remainder'
        Name of the zero remainder function.

    Methods
    ----------
    __init__
        It initializes the zero remainder function.
    """
    def __init__(self, name='zero_remainder', *args, **kwargs):
        """
        The initialization method of the zero remainder function.

        It initializes a zero remainder function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'zero_remainder'
            Name of the zero remainder function.

        Returns
        ----------
        object
            The zero remainder function object.
        """
        super().__init__(name=name, c=0.0, *args, **kwargs)


class one_remainder(constant_remainder):
    r"""
    The one remainder function.

    As a special case of constant remainder, the zero remainder returns constant one vector as the remainder term.
    This class inherits from the constant remainder class.

    ...

    Notes
    ----------
    The zero remainder function $\pi: {R}^m \to {R}^n$ just projects all inputs to a constant one vector, i.e.,
    $$
        \begin{equation}
            \pi(\mathbf{x}) =  \mathbf{1}^n \in {R}^n.
        \end{equation}
    $$

    By default, the remainder term will also be processed with the optional activation functions.

    Attributes
    ----------
    name: str, default = 'one_remainder'
        Name of the one remainder function.

    Methods
    ----------
    __init__
        It initializes the zero remainder function.
    """
    def __init__(self, name='one_remainder', *args, **kwargs):
        """
        The initialization method of the one remainder function.

        It initializes a one remainder function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'one_remainder'
            Name of the one remainder function.

        Returns
        ----------
        object
            The one remainder function object.
        """
        super().__init__(name=name, c=1.0, *args, **kwargs)


class identity_remainder(remainder):
    r"""
    The identity remainder function.

    It calculates the identity remainder, and returns input data as the remainders.
    This class inherits from the base remainder class (i.e., the remainder class in the module directory).

    ...

    Notes
    ----------
    The identity remainder function $\pi: {R}^m \to {R}^n$ just projects all inputs to the inputs themselves, i.e.,
    $$
        \begin{equation}
            \pi(\mathbf{x}) = \mathbf{x} \in {R}^n,
        \end{equation}
    $$
    where the input and output dimensions need to be equal, i.e., $m = n$.

    By default, the remainder term will also be processed with the optional activation functions.

    Attributes
    ----------
    name: str, default = 'identity_remainder'
        Name of the identity remainder function.

    Methods
    ----------
    __init__
        It initializes the identity remainder function.

    forward
        It implements the abstract forward method declared in the base remainder class.
    """
    def __init__(self, name='identity_remainder', *args, **kwargs):
        """
        The initialization method of the identity remainder function.

        It initializes an identity remainder function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'identity_remainder'
            Name of the identity remainder function.

        Returns
        ----------
        object
            The identity remainder function object.
        """
        super().__init__(name=name, *args, **kwargs)

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
       The forward method of the identity remainder function.

       The identity remainder function $\pi: {R}^m \to {R}^n$ just projects all inputs to the inputs themselves, i.e.,
        $$
            \begin{equation}
                \pi(\mathbf{x}) = \mathbf{x} \in {R}^n,
            \end{equation}
        $$
        where the input and output dimensions need to be equal, i.e., $m = n$.

        By default, the remainder term will also be processed with the optional activation functions.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            Device to calculate the remainder function.

        Returns
        ----------
        torch.Tensor
            The remainder term of length $n$.
        """
        return self.activation(x=x, device=device)


class linear_remainder(remainder):
    r"""
    The linear remainder function.

    It calculates the linear remainder, and returns remainder term of certain length.
    The identity remainder function can only be applied when the input and output dimensions are equal.
    For more general cases where the input and output dimensions are different, the linear remainder function can be used.
    This class inherits from the base remainder class (i.e., the remainder class in the module directory).

    ...

    Notes
    ----------
    The linear remainder function $\pi: {R}^m \to {R}^n$ just projects all inputs to themselves subject to linear
    transformations, i.e.,
    $$
        \begin{equation}
            \pi(\mathbf{x}) = \mathbf{x} \mathbf{W} \in {R}^n,
        \end{equation}
    $$
    where the learnable parameter matrix $\mathbf{W} \in R^{m \times n}$ is used for vector dimension adjustment.

    By default, the remainder term will also be processed with the optional activation functions.

    Attributes
    ----------
    name: str, default = 'linear_remainder'
        Name of the remainder function.

    Methods
    ----------
    __init__
        It initializes the linear remainder function.

    forward
        It implements the abstract forward method declared in the base remainder class.
    """
    def __init__(self, name: str = 'linear_remainder', require_parameters: bool = True, enable_bias: bool = False, *args, **kwargs):
        """
        The initialization method of the linear remainder function.

        It initializes a linear remainder function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'linear_remainder'
            Name of the linear remainder function.
        require_parameters: bool, default = False
            Boolean tag of whether the function requires parameters.
        enable_bias: bool, default = False
            Boolean tag of whether the bias is enabled or not.

        Returns
        ----------
        object
            The linear remainder function object.
        """
        super().__init__(name=name, require_parameters=require_parameters, enable_bias=enable_bias, *args, **kwargs)

    def forward(self, x: torch.Tensor, w: torch.nn.Parameter, b: torch.nn.Parameter = None, device='cpu', *args, **kwargs):
        r"""
        The forward method of the linear remainder function.

        The linear remainder function $\pi: {R}^m \to {R}^n$ just projects all inputs to themselves subject to linear
        transformations, i.e.,
        $$
            \begin{equation}
                \pi(\mathbf{x}) = \mathbf{x} \mathbf{W} \in {R}^n,
            \end{equation}
        $$
        where the learnable parameter matrix $\mathbf{W} \in R^{m \times n}$ is used for vector dimension adjustment.

        By default, the remainder term will also be processed with the optional activation functions.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        w: torch.nn.Parameter
            The linear transformation parameter.
        b: torch.nn.Parameter, default = None
            The linear transformation bias parameter. It will not be None if attribute "enable_bias" is assigned with "True" value at initialization.
        device: str, default = 'cpu'
            Device to calculate the remainder function.

        Returns
        ----------
        torch.Tensor
            The remainder term of the input.
        """

        if w is not None:
            x = F.linear(x, w, bias=b)
        return self.activation(x=x, device=device)