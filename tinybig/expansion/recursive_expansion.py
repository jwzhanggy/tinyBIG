# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Recursive data expansion functions.

This module contains the polynomial data expansion functions with recursive definitions,
including bspline_expansion, chebyshev_expansion and jacobi_expansion.
"""

import torch.nn

from tinybig.expansion import transformation


#####################################################
# Expansions defined with recursively defined basis #
#####################################################

class bspline_expansion(transformation):
    r"""
    The bspline data expansion function.

    It performs the bspline expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its bspline expansion of degree $d$ will be
    $$
        \begin{equation}
            \kappa (\mathbf{x} | d) = \left[ B_{0,d}(\mathbf{x}), B_{1,d}(\mathbf{x}), \cdots, B_{t-1,d}(\mathbf{x}) \right] \in {R}^D,
        \end{equation}
    $$
    where $B_{i,d}(\mathbf{x})$ denotes the bspline expansion polynomials of $\mathbf{x}$ of degree $d$
    within the range $[x_i, x_{i+d+1})$. The output dimension will be $D = m (t + d)$.

    As to the specific representations of bspline polynomials, they can be defined recursively based on the
    lower-degree terms according to the following equations:

    (1) **Base B-splines with degree $d = 0$:**
    $$
        \begin{equation}
            \{ B\_{0,0}(x), B\_{1,0}(x), \cdots, B\_{t-1,0}(x) \},
        \end{equation}
    $$
        where
    $$
        \begin{equation}
            B\_{i,0}(x) = \begin{cases}
            1, &\text{if } x_i \le x < x_{i+1};\\\\
            0, &\text{otherwise}.
            \end{cases}
        \end{equation}
    $$

    (2) **Higher-degree B-splines with $d > 0$:**
    $$
        \begin{equation}
            \{ B\_{0,d}(x), B\_{1,d}(x), \cdots, B\_{t-1,d}(x) \},
        \end{equation}
    $$
    where
    $$
        \begin{equation}
            B\_{i,d}(x) = \frac{x - x_i}{x_{i+d} - x_i} B\_{i, d-1}(x) + \frac{x_{i+d+1} - x}{x_{i+d+1} - x_{i+1}} B\_{i+1, d-1}(x).
        \end{equation}
    $$
    According to the representations, term $B_{i,d}(x)$ recursively defined above will have non-zero outputs
    if and only if the inputs lie within the value range $x_i \le x < x_{i+d+1}$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the bspline expansion function.

    Attributes
    ----------
    name: str, default = 'bspline_expansion'
        Name of the expansion function.
    grid_range: tuple, default = (-1, 1)
        The input value range for defining the grid.
    t: int, default = 5
        The interval number divided by the knots in bspline.
    d: int, default = 3
        The degree of the bspline expansion.
    grid: torch.Tensor, default = None
        The grid representing relationships between lower-order bspline polynomials with high-order ones.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    initialize_grid
        It initializes the grid defining the relationships between lower-order bspline polynomials with high-order ones.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='bspline_expansion', grid_range=(-1, 1), t=5, d=3, *args, **kwargs):
        r"""
        The initialization method of bspline expansion function.

        It initializes a bspline expansion object based on the input function name and parameters.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'bspline_expansion'
            The name of the bspline expansion function.
        grid_range: tuple, default = (-1, 1)
            The input value range for defining the grid.
        t: int, default = 5
            The interval number divided by the knots in bspline.
        d: int, default = 3
            The degree of the bspline expansion.

        Returns
        -------
        object
            The bspline expansion function object.
        """
        super().__init__(name=name, *args, **kwargs)
        self.grid_range = grid_range
        self.t = t
        self.d = d
        self.grid = None

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the bspline expansion function, the expansion space dimension is determined by m, and parameters t and d,
        which can be represented as:

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

    def initialize_grid(self, m: int, device='cpu', grid_range: tuple | list = None, t: int = None, d: int = None):
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
            The grid tensor of shape (m, t+d) denoting the lower-order and high-order bspline polynomial relationships.
        """
        grid_range = grid_range if grid_range is not None else self.grid_range
        t = t if t is not None else self.t
        d = d if d is not None else self.d

        h = (grid_range[1] - grid_range[0]) / t
        self.grid = torch.Tensor((torch.arange(-d, t + d + 1) * h + grid_range[0])
                                 .expand(m, -1).contiguous()).to(device)

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the bspline data expansion of the input data and returns the expansion result
        according to the following equation:
        $$
        \begin{equation}
            \kappa (\mathbf{x} | d) = \left[ B_{0,d}(\mathbf{x}), B_{1,d}(\mathbf{x}), \cdots, B_{t-1,d}(\mathbf{x}) \right] \in {R}^D,
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
        if self.grid is None:
            self.initialize_grid(m=x.size(1), device=device, *args, **kwargs)
        assert x.dim() == 2
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.d + 1):
            bases = (((x - self.grid[:, : -(k + 1)]) / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)]) * bases[:, :, :-1]) +
                     ((self.grid[:, k + 1:] - x) / (self.grid[:, k + 1:] - self.grid[:, 1:(-k)]) * bases[:, :, 1:]))
        expansion = bases.contiguous().view(x.size(0), -1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class chebyshev_expansion(transformation):
    r"""
    The chebyshev data expansion function.

    It performs the chebyshev expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its chebyshev expansion up to degree $d$ can be represented as
    $$
        \begin{equation}
            \kappa(\mathbf{x} | d) = \left[ T_1(\mathbf{x}), T_2(\mathbf{x}) \cdots, T_d(\mathbf{x}) \right] \in {R}^D,
        \end{equation}
    $$
    where $T_d(\mathbf{x})$ denotes the chebyshev expansion polynomial of $\mathbf{x}$ of degree $d$.
    The output dimension of chebyshev expansion will be $D = m d$.

    As to the specific representations of chebyshev polynomials, they can be defined recursively based on the
    lower-degree terms according to the following equations:

    (1) **Base chebyshev polynomial with degree $d=0$ and $d=1$:**
    $$
        \begin{equation}
            T_0(x) = 1 \text{, and } T_1(x) = x.
        \end{equation}
    $$

    (2) **Higher-degree chebyshev polynomial with $d \ge 2$:**
    $$
        \begin{equation}
            T_d(x) = 2x \cdot T_{d-1}(x) - T_{d-2}(x).
        \end{equation}
    $$

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the chebyshev expansion function.

    Attributes
    ----------
    name: str, default = 'chebyshev_expansion'
        Name of the expansion function.
    d: int, default = 2
        Degree of chebyshev expansion.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='chebyshev_polynomial_expansion', d=5, *args, **kwargs):
        r"""
        The initialization method of chebyshev expansion function.

        It initializes a chebyshev expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'chebyshev_polynomial_expansion'
            The name of the chebyshev expansion function.
        d: int, default = 5
            The degree of the chebyshev expansion function.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the chebyshev expansion function, the expansion space dimension is determined by m and d,
        which can be represented as:

        $$ D = m d. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the chebyshev data expansion of the input data and returns the expansion result
        according to the following equation:
        $$
        \begin{equation}
            \kappa(\mathbf{x} | d) = \left[ T_1(\mathbf{x}), T_2(\mathbf{x}) \cdots, T_d(\mathbf{x}) \right] \in {R}^D.
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
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d+1]).to(device)
        if self.d > 0:
            expansion[:,:,1] = x
        for n in range(2, self.d+1):
            expansion[:, :, n] = 2 * x * expansion[:, :, n-1].clone() - expansion[:, :, n-2].clone()
        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)
        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class jacobi_expansion(transformation):
    r"""
    The jacobi data expansion function.

    It performs the jacobi expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its jacobi expansion up to degree $d$ can be represented as
    $$
        \begin{equation}
            \kappa(\mathbf{x} | d) = \left[ P_1^{(\alpha, \beta)}(\mathbf{x}), P_2^{(\alpha, \beta)}(\mathbf{x}),  \cdots, P_d^{(\alpha, \beta)}(\mathbf{x})\right] \in {R}^D,
        \end{equation}
    $$
    where $P_d^{(\alpha, \beta)}(\mathbf{x})$ denotes the jacobi expansion polynomial of $\mathbf{x}$ degree $d$.
    The output dimension of jacobi expansion will be $D = m d$.

    As to the specific representations of jacobi polynomials, they can be defined recursively based on the
    lower-degree terms according to the following equations:

    (1) **Base jacobi polynomials with degree $d=0$, $d=1$ and $d=2$:**
    $$
        \begin{align}
            P\_0^{(\alpha, \beta)}(x) &= 1,\\\\
            P\_1^{(\alpha, \beta)}(x) &= (\alpha + 1) + (\alpha + \beta + 2) \frac{(x-1)}{2},\\\\
            P\_2^{(\alpha, \beta)}(x) &= \frac{(\alpha+1)(\alpha+2)}{2} + (\alpha+2)(\alpha+\beta+3) \frac{x-1}{2} + \frac{(\alpha + \beta + 3)(\alpha + \beta + 4)}{2} \left( \frac{x-1}{2} \right)^2.
        \end{align}
    $$

    (2) **Higher-degree jacobi polynomials with $d \ge 2$:**
    $$
        \begin{align}
            P\_d^{(\alpha, \beta)}(x) &= \frac{(2d + \alpha + \beta -1) \left[ (2d + \alpha + \beta)(2d + \alpha + \beta -2) x + (\alpha^2 - \beta^2) \right]}{2d(d + \alpha + \beta)(2d + \alpha + \beta - 2) } P\_{d-1}^{(\alpha, \beta)}(x)\\\\
            & - \frac{2(d+\alpha-1)(d+\beta-1)(2d+\alpha+\beta)}{2d(d + \alpha + \beta)(2d + \alpha + \beta - 2) }P\_{d-2}^{(\alpha, \beta)}(x).
        \end{align}
    $$

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the jacobi expansion function.

    Attributes
    ----------
    name: str, default = 'jacobi_polynomial_expansion'
        Name of the expansion function.
    d: int, default = 5
        Degree of jacobi expansion.
    alpha: float, default = 1.0
        Parameter of jacobi polynomial representation.
    beta: float, default = 1.0
        Parameter of jacobi polynomial representation.
    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='jacobi_polynomial_expansion', d=5, alpha=1.0, beta=1.0, *args, **kwargs):
        r"""
        The initialization method of jacobi expansion function.

        It initializes a jacobi expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'jacobi_polynomial_expansion'
            The name of the jacobi expansion function.
        d: int, default = 5
            The degree of the jacobi expansion function.
        alpha: float, default = 1.0
            Parameter of jacobi polynomial representation.
        beta: float, default = 1.0
            Parameter of jacobi polynomial representation.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d
        self.alpha = alpha
        self.beta = beta

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the jacobi expansion function, the expansion space dimension is determined by m and d,
        which can be represented as:

        $$ D = m d. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the jacobi data expansion of the input data and returns the expansion result
        according to the following equation:
        $$
            \begin{equation}
                \kappa(\mathbf{x} | d) = \left[ P_1^{(\alpha, \beta)}(\mathbf{x}), P_2^{(\alpha, \beta)}(\mathbf{x}),  \cdots, P_d^{(\alpha, \beta)}(\mathbf{x})\right] \in {R}^D.
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
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d+1]).to(device)
        if self.d > 0:
            expansion[:,:,1] = ((self.alpha-self.beta) + (self.alpha+self.beta+2) * x) / 2
        for n in range(2, self.d+1):
            coeff_1 = 2*n*(n+self.alpha+self.beta)*(2*n+self.alpha+self.beta-2)
            coeff_2 = (2*n+self.alpha+self.beta-1)*(2*n+self.alpha+self.beta)*(2*n+self.alpha+self.beta-2)
            coeff_3 = (2*n+self.alpha+self.beta-1)*(self.alpha**2-self.beta**2)
            coeff_4 = 2*(n+self.alpha-1)*(n+self.beta-1)*(2*n+self.alpha+self.beta)
            expansion[:,:,n] = ((coeff_2/coeff_1)*x + coeff_3/coeff_1)*expansion[:,:,n-1].clone() - (coeff_4/coeff_1)*expansion[:,:,n-2].clone()
        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)

