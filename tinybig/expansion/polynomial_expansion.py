# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Polynomial data expansion functions.

This module contains the polynomial data expansion functions,
including taylor_expansion and fourier_expansion.
"""


import warnings
import numpy as np
import torch.nn

from tinybig.expansion import transformation

###################################################
# Expansions defined with closed-form polynomials #
###################################################


class taylor_expansion(transformation):
    r"""
    The taylor's data expansion function.

    It performs the taylor's expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its taylor's expansion will be
    $$
        \begin{equation}
            \kappa (\mathbf{x} | d) = [P\_1(\mathbf{x}), P\_2(\mathbf{x}), \cdots, P\_d(\mathbf{x}) ] \in {R}^D.
        \end{equation}
    $$
    where $P_d(\mathbf{x})$ denotes the taylor's expansion of $\mathbf{x}$ of degree $d$. The output dimension will then be $D = \sum_{i=1}^d m^i$.

    Specifically, $P_d(\mathbf{x})$ can be recursively defined as follows:
    $$
        \begin{align}
            P\_0(\mathbf{x}) &= [1] \in {R}^{1},\\\\
            P\_1(\mathbf{x}) &= [x\_1, x\_2, \cdots, x\_m] \in {R}^{m},\\\\
            P\_d(\mathbf{x}) &= P\_1(\mathbf{x}) \otimes P\_{d-1}(\mathbf{x}) \text{, for } \forall d \ge 2.
        \end{align}
    $$

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the taylor's expansion function.

    Attributes
    ----------
    name: str, default = 'taylor_expansion'
        Name of the expansion function.
    d: int, default = 2
        Degree of taylor's expansion.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """

    def __init__(self, name='taylor_expansion', d=2, *args, **kwargs):
        r"""
        The initialization method of taylor's expansion function.

        It initializes a taylor's expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'taylor_expansion'
            The name of the taylor's expansion function.
        d: int, default = 2
            The max degree of the taylor's expansion.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the taylor's expansion function, the expansion space dimension is determined by both m and d,
        which can be represented as:

        $$ D = \sum_{i=1}^d m^i. $$

        Notes
        ----------
        Taylor's expansion function will increase the expansion dimension exponentially and the degree parameter $d$
        is usually set with a small number. When the expansion dimension $D > 10^7$ (i.e., more than 10 million),
        the function will raise a warning reminder.

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        D = sum([m**i for i in range(1, self.d+1)])
        if D > 10**7:
            warnings.warn('You have expanded the input data to a very high-dimensional representation, '
                          'with more than 10M features per instance...', UserWarning)
        return D

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the taylor's data expansion of the input data and returns the expansion result
        according to the following equation:
        $$
        \begin{equation}
            \kappa (\mathbf{x} | d) = [P\_1(\mathbf{x}), P\_2(\mathbf{x}), \cdots, P\_d(\mathbf{x}) ] \in {R}^D.
        \end{equation}
        $$


        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """

        x = self.pre_process(x=x, device=device)
        x_powers = torch.ones(size=[x.size(0), 1]).to(device)
        expansion = torch.Tensor([]).to(device)

        for i in range(1, self.d+1):
            x_powers = torch.einsum('ba,bc->bac', x_powers.clone(), x).view(x_powers.size(0), x_powers.size(1)*x.size(1))
            expansion = torch.cat((expansion, x_powers), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class fourier_expansion(transformation):
    r"""
    The fourier data expansion function.

    It performs the fourier expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, based on the parameters $P$ and $N$, its fourier expansion will be
    $$
        \begin{equation}
            \kappa (\mathbf{x} | P, N) = \left[ \cos (2\pi \frac{1}{P} \mathbf{x} ), \sin(2\pi \frac{1}{P} \mathbf{x} ), \cdots, \cos(2\pi \frac{N}{P} \mathbf{x} ), \sin(2\pi \frac{N}{P} \mathbf{x} ) \right] \in {R}^D,
        \end{equation}
    $$
    where the output dimension $D = 2 m N$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the fourier expansion function.

    Attributes
    ----------
    name: str, default = 'fourier_expansion'
        Name of the expansion function.
    P: int, default = 10
        The period parameter of the expansion.
    N: int, default = 5
        The harmonic number of the expansion.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='fourier_expansion', P=10, N=5, *args, **kwargs):
        r"""
        The initialization method of fourier expansion function.

        It initializes a fourier expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'fourier_expansion'
            The name of the fourier expansion function.
        P: int, default = 10
            The period parameter of the expansion.
        N: int, default = 5
            The harmonic number of the expansion.
        """
        super().__init__(name=name, *args, **kwargs)
        self.P = P
        self.N = N

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the fourier expansion function, the expansion space dimension is determined by both m and N,
        which can be represented as:

        $$ D = 2 m N. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m * self.N * 2

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the fourier data expansion of the input data and returns the expansion result
        according to the following equation:
        $$
        \begin{equation}
            \kappa (\mathbf{x} | P, N) = \left[ \cos (2\pi \frac{1}{P} \mathbf{x} ), \sin(2\pi \frac{1}{P} \mathbf{x} ), \cdots, \cos(2\pi \frac{N}{P} \mathbf{x} ), \sin(2\pi \frac{N}{P} \mathbf{x} ) \right] \in {R}^D,
        \end{equation}
        $$

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        x = self.pre_process(x=x, device=device)
        expansion = torch.Tensor([]).to(device)
        for n in range(1, self.N+1):
            cos = torch.cos(2 * np.pi * (n / self.P) * x)
            sin = torch.sin(2 * np.pi * (n / self.P) * x)
            expansion = torch.cat((expansion, cos, sin), dim=1)
        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


