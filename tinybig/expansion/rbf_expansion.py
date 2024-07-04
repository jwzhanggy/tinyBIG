# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
rfb data expansion functions.

This module contains the rfb data expansion functions,
including gaussian_rbf_expansion and inverse_quadratic_rbf_expansion.
"""

import torch.nn

from tinybig.expansion import transformation


#########################################################
# Expansions defined with RBF for given base fix points #
#########################################################

class gaussian_rbf_expansion(transformation):
    r"""
    The gaussian rbf data expansion function.

    It performs the gaussian rbf expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its gaussian rbf expansion with $d$ fixed points can be represented as follows:
    $$
    \begin{equation}
        \kappa(\mathbf{x}) = {\varphi} (\mathbf{x} | \mathbf{c}) = \left[ \varphi (\mathbf{x} | c_1), \varphi (\mathbf{x} | c_2), \cdots, \varphi (\mathbf{x} | c_d) \right] \in {R}^D,
    \end{equation}
    $$
    where the sub-vector element ${\varphi} (x | \mathbf{c})$ can be defined as follows:
    $$
        \begin{equation}
            {\varphi} (x | \mathbf{c}) = \left[ \varphi (x | c_1), \varphi (x | c_2), \cdots, \varphi (x | c_d) \right] \in {R}^d.
        \end{equation}
    $$
    and value $\varphi (x | c)$ is defined as:
    $$
        \begin{equation}
            \varphi (x | c) = \exp(-(\epsilon (x - c) )^2).
        \end{equation}
    $$

    For gaussian rbf expansion, its output expansion dimensions will be $D = md$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the gaussian rbf expansion function.

    Attributes
    ----------
    name: str, default = 'gaussian_rbf_expansion'
        Name of the expansion function.
    base_range: tuple | list, default = (-1, 1)
        Input value range.
    num_interval: int, default = 10
        Number of intervals partitioned by the fixed points.
    epsilon: float, default = 1.0
        The rbf function parameter.
    base: torch.Tensor
        The partition of value range into intervals, i.e., the vector $\mathbf{c}$ in the above equation.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='gaussian_rbf_expansion', base_range=(-1, 1), num_interval=10, epsilon=1.0, base=None, *args, **kwargs):
        r"""
        The initialization method of the gaussian rbf expansion function.

        It initializes a gaussian rbf expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'gaussian_rbf_expansion'
            The name of the gaussian rbf expansion function.
        base_range: tuple | list, default = (-1, 1)
            Input value range.
        num_interval: int, default = 10
            Number of intervals partitioned by the fixed points.
        epsilon: float, default = 1.0
            The rbf function parameter.
        base: Tensor, default = None
            The partition of value range into intervals, i.e., the vector $\mathbf{c}$ in the above equation.

        """
        super().__init__(name=name, *args, **kwargs)
        self.betaase_range = base_range
        self.num_interval = num_interval
        self.epsilon = epsilon
        self.base = base

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the gaussian rbf expansion function, the expansion space dimension will be
        $$ D = m d, $$
        where $d$ denotes the number of intervals of the input value range.

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m * self.num_interval

    def initialize_base(self, device='cpu', base_range=None, num_interval=None):
        r"""
        The fixed point base initialization method.

        It initializes the fixed point base tensor, which partitions the value range into equal-length intervals.
        The initialized base tensor corresponds to the fixed point vector $\mathbf{c}$ mentioned in the above equation.

        Parameters
        ----------
        device: str, default = 'cpu'
            The device to host the base tensor.
        base_range: tuple | list, default = None
            Input value range.
        num_interval: int, default = None
            Number of intervals partitioned by the fixed points.

        Returns
        -------
        torch.Tensor
            The fixed point base tensor.
        """
        base_range = base_range if base_range is not None else self.betaase_range
        num_interval = num_interval if num_interval is not None else self.num_interval
        self.base = torch.Tensor(torch.linspace(base_range[0], base_range[1], num_interval)).to(device)

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the gaussian rbf data expansion of the input data and returns the expansion result as
        $$
            \begin{equation}
                \kappa(\mathbf{x}) = {\varphi} (\mathbf{x} | \mathbf{c}) = \left[ \varphi (\mathbf{x} | c_1), \varphi (\mathbf{x} | c_2), \cdots, \varphi (\mathbf{x} | c_d) \right] \in {R}^D,
            \end{equation}
        $$
        where vector $\mathbf{c}$ is the fixed point base tensor initialized above.


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
        if self.base is None:
            self.initialize_base(device=device, *args, **kwargs)
        assert x.dim() == 2
        expansion = torch.exp(-((x[..., None] - self.base) * self.epsilon) ** 2).view(x.size(0), -1)
        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class inverse_quadratic_rbf_expansion(gaussian_rbf_expansion):
    r"""
    The inverse quadratic rbf data expansion function.

    It performs the inverse quadratic rbf expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its inverse quadratic rbf expansion with $d$ fixed points can be represented as follows:
    $$
    \begin{equation}
        \kappa(\mathbf{x}) = {\varphi} (\mathbf{x} | \mathbf{c}) = \left[ \varphi (\mathbf{x} | c_1), \varphi (\mathbf{x} | c_2), \cdots, \varphi (\mathbf{x} | c_d) \right] \in {R}^D,
    \end{equation}
    $$
    where the sub-vector element ${\varphi} (x | \mathbf{c})$ can be defined as follows:
    $$
        \begin{equation}
            {\varphi} (x | \mathbf{c}) = \left[ \varphi (x | c_1), \varphi (x | c_2), \cdots, \varphi (x | c_d) \right] \in {R}^d.
        \end{equation}
    $$
    and value $\varphi (x | c)$ is defined as:
    $$
        \begin{equation}
            \varphi (x | c) = \frac{1}{1 + (\epsilon (x - c))^2}.
        \end{equation}
    $$

    For inverse quadratic rbf expansion, its output expansion dimensions will be $D = md$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the inverse quadratic rbf expansion function.

    Attributes
    ----------
    name: str, default = 'inverse_quadratic_rbf_expansion'
        Name of the expansion function.
    base_range: tuple | list, default = (-1, 1)
        Input value range.
    num_interval: int, default = 10
        Number of intervals of the input value range.
    epsilon: float, default = 1.0
        The rbf function parameter.
    base: torch.Tensor
        The partition of value range into intervals, i.e., the vector $\mathbf{c}$ in the above equation.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='inverse_quadratic_rbf', base_range=(-1, 1), num_interval=10, epsilon=1.0, base=None, *args, **kwargs):
        r"""
        The initialization method of the inverse quadratic rbf expansion function.

        It initializes an inverse quadratic rbf expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'inverse_quadratic_rbf'
            The name of the inverse quadratic rbf expansion function.
        base_range: tuple | list, default = (-1, 1)
            Input value range.
        num_interval: int, default = 10
            Number of intervals partitioned by the fixed points.
        epsilon: float, default = 1.0
            The rbf function parameter.
        base: Tensor, default = None
            The partition of value range into intervals, i.e., the vector $\mathbf{c}$ in the above equation.

        """
        super().__init__(name=name, base_range=base_range, num_interval=num_interval, epsilon=epsilon, base=base, *args, **kwargs)

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the inverse quadratic rbf data expansion of the input data and returns the expansion result as
        $$
            \begin{equation}
                \kappa(\mathbf{x}) = {\varphi} (\mathbf{x} | \mathbf{c}) = \left[ \varphi (\mathbf{x} | c_1), \varphi (\mathbf{x} | c_2), \cdots, \varphi (\mathbf{x} | c_d) \right] \in {R}^D,
            \end{equation}
        $$
        where vector $\mathbf{c}$ is the fixed point base tensor initialized above.


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
        if self.base is None:
            self.initialize_base(device=device, *args, **kwargs)
        assert x.dim() == 2
        expansion = (1/(1+((x[..., None] - self.base) * self.epsilon) ** 2)).view(x.size(0), -1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)

