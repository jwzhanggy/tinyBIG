# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Complementary expansion based remainder function.

This module contains the complementary expansion based remainder function defined based on bspines,
i.e., the bspline_remainder.
"""

import torch
import torch.nn.functional as F

from tinybig.remainder import remainder


##############################
# Expansion based Remainders #
##############################


class bspline_remainder(remainder):
    r"""
    The complementary expansion based remainder function defined with bsplines.

    It performs the bspline expansion based remainder function.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    Formally, the complementary bspline expansion based remainder function can be defined as follows:
    $$
        \begin{equation}
            \pi(\mathbf{x} | \mathbf{w}', \mathbf{W}) = \left(\left\langle \kappa'(\mathbf{x}), \psi'(\mathbf{w}') \right\rangle + \pi'(\mathbf{x}) \right) \mathbf{W} \in R^n,
        \end{equation}
    $$
    where $\mathbf{w}'$ is the learnable parameter of the remainder function.
    Term $\kappa'(\mathbf{x})$ is defined as the bspline data expansion function, $\psi'(\mathbf{w}')$ is the identity
    parameter reconciliation function and $\pi'(\mathbf{x})$ is the zero remainder function.

    (1) For input vector $\mathbf{x} \in R^m$, its bspline expansion of degree $d$ used above is defined as
    $$
        \begin{equation}
            \kappa' (\mathbf{x} | d) = \left[ B_{0,d}(\mathbf{x}), B_{1,d}(\mathbf{x}), \cdots, B_{t-1,d}(\mathbf{x}) \right] \in {R}^D,
        \end{equation}
    $$
    where $B_{i,d}(\mathbf{x})$ denotes the bspline expansion polynomials of $\mathbf{x}$ of degree $d$
    within the range $[x_i, x_{i+d+1})$. The output dimension will be $D = m (t + d)$.

    (2) Meanwhile, for the parameter $\mathbf{w}' \in R^{n \times D}$, its identity reconciliation function is defined as
    $$
        \begin{equation}
            \psi'(\mathbf{w}') = \mathbf{w}' \in R^{m \times D}.
        \end{equation}
    $$

    (3) As to the remainder function used above, we will use the zero remainder by default, i.e.,
    $$
        \begin{equation}
            \pi'(\mathbf{x}) = \mathbf{0} \in R^{m}.
        \end{equation}
    $$

    In addition to the bspline expansions, this remainder function will apply a linear transformation to the output
    to adjust the output dimensions from m to n, where the learnable parameter matrix $\mathbf{W} \in R^{m \times n}$
    is used for vector dimension adjustment.

    Attributes
    ----------
    name: str, default = 'bspline_remainder'
        Name of the remainder function.
    grid_range: tuple, default = (-1, 1)
        The input value range for defining the grid.
    t: int, default = 5
        The interval number divided by the knots in bspline.
    d: int, default = 3
        The degree of the bspline expansion.
    grid: torch.Tensor, default = None
        The grid representing relationships between lower-order bspline polynomials with high-order ones.
    reconciled_parameters: torch.nn.Parameter, default = None
        The parameters of the remainder function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    initialize_reconciled_parameters
        It initializes the learnable parameters of the remainder function.

    initialize_grid
        It initializes the grid defining the relationships between lower-order bspline polynomials with high-order ones.

    forward
        It implements the abstract forward method declared in the base remainder class.

    """
    def __init__(self, name='bspline_remainder', grid_range=(-1, 1), t=5, d=3,
                 require_parameters: bool = True, enable_bias: bool = False, *args, **kwargs):
        r"""
        The initialization method of bspline expansion based remainder function.

        It initializes a bspline expansion based reminder function object based on the input function name and parameters.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'bspline_remainder'
            The name of the bspline expansion based remainder function.
        grid_range: tuple, default = (-1, 1)
            The input value range for defining the grid.
        t: int, default = 5
            The interval number divided by the knots in bspline.
        d: int, default = 3
            The degree of the bspline expansion.

        Returns
        -------
        object
            The bspline expansion based remainder function object.
        """
        super().__init__(name=name, require_parameters=require_parameters, enable_bias=enable_bias, *args, **kwargs)
        self.grid_range = grid_range
        self.t = t
        self.d = d
        self.grid = None
        self.reconciled_parameters = None

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the bspline expansion function used in the remainder function, the expansion space dimension is
        determined by m, and parameters t and d, which can be represented as:

        $$ D = m (t + d). $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m * (self.t + self.d)

    def initialize_reconciled_parameters(self, m: int, D: int, device='cpu'):
        """
        The parameter initialization method.

        It initializes the learnable parameters of shape (m, D) to be used in the remainder function.
        In the current remainder function, the identity parameter reconciliation function is used.

        Parameters
        ----------
        m: int
            The dimension of the input space.
        D: int
            The dimension of the intermediate expansion space.
        device: str, default = 'cpu'
            Device for hosting the parameter.

        Returns
        -------
        None
            The function initializes the parameter and doesn't have return values.
        """
        self.reconciled_parameters = torch.nn.Parameter(torch.rand(m, D)).to(device)
        torch.nn.init.xavier_uniform_(self.reconciled_parameters)

    def initialize_grid(self, m: int, grid_range=None, t=None, d=None, device='cpu', *args, **kwargs):
        """
        The grid initialization method.

        It initializes the grid defining the relationships between lower-order bspline polynomials with high-order ones.
        The grid enables faster calculation of the high-order bspline polynomials, which is defined by the parameters
        grid value range "grid_range", interval number $t$ and polynomial degree $d$.

        The grid is defined as a tensor of shape (m, t+d).

        Parameters
        ----------
        m: int
            The dimension of the input space.
        device: str, default = 'cpu'
            The device to host the grid.
        grid_range: tuple | list, default = None
            The customized grid value range.
        t: int, default = None
            The interval number divided by the knots in bspline.
        d: int, default = None
            The degree of the bspline expansion.

        Returns
        -------
        torch.Tensor
            The function returns a grid tensor of shape (m, t+d) denoting the lower-order and high-order
            bspline polynomial relationships.
        """
        grid_range = grid_range if grid_range is not None else self.grid_range
        t = t if t is not None else self.t
        d = d if d is not None else self.d

        h = (grid_range[1] - grid_range[0]) / t
        self.grid = torch.Tensor((torch.arange(-d, t + d + 1) * h + grid_range[0]).expand(m, -1).contiguous()).to(device)

    def forward(self, x: torch.Tensor, w: torch.nn.Parameter = None, b: torch.nn.Parameter = None, device='cpu', *args, **kwargs):
        r"""
        The forward method of the bspline expansion based remainder function.

        The complementary bspline expansion based remainder function calculates the remainder term as follows:
        $$
            \begin{equation}
                \pi(\mathbf{x} | \mathbf{w}', \mathbf{W}) = \left(\left\langle \kappa'(\mathbf{x}), \psi'(\mathbf{w}') \right\rangle + \pi'(\mathbf{x}) \right) \mathbf{W} \in R^n,
            \end{equation}
        $$
        where $\kappa'$ denotes the bspline expansion function, $\psi'$ is the identity parameter reconciliation function,
        and $\pi'$ denotes the zero remainder function. The learnable parameter matrix $\mathbf{W} \in R^{m \times n}$
        is used for vector dimension adjustment.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        w: torch.nn.Parameter, default = None
            The linear transformation parameter. It will not be None if attribute "require_parameters" is assigned with "True" value at initialization.
        b: torch.nn.Parameter, default = None
            The linear transformation bias parameter. It will not be None if attribute "enable_bias" is assigned with "True" value at initialization.
        device: str, default = 'cpu'
            Device to calculate the remainder function.

        Returns
        ----------
        torch.Tensor
            The remainder term of the input.
        """
        if self.grid is None:
            self.initialize_grid(device=device, *args, **kwargs)
        if self.require_parameters and self.reconciled_parameters is None:
            self.initialize_reconciled_parameters(m=x.size(1), D=self.calculate_D(x.size(1)), device=device)

        assert x.dim() == 2
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.d + 1):
            bases = (((x - self.grid[:, : -(k + 1)]) / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)]) * bases[:, :, :-1]) +
                     ((self.grid[:, k + 1:] - x) / (self.grid[:, k + 1:] - self.grid[:, 1:(-k)]) * bases[:, :, 1:]))

        assert bases.size() == (x.size(0), x.size(1), self.t + self.d)
        # [B, m] -> [B, D] x [m, D]^T -> [B, m]
        x = F.linear(bases.contiguous().view(x.size(0), -1), self.reconciled_parameters)
        # [B, m] -> [B, n]
        x = F.linear(x, w, bias=b)
        return self.activation(x=x, device=device)