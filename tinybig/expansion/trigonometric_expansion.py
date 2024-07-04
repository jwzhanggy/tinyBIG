# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
trigonometric data expansion functions.

This module contains the trigonometric data expansion functions,
including hyperbolic_expansion, arc_hyperbolic_expansion, trigonometric_expansion and arc_trigonometric_expansion.
"""

import torch.nn

from tinybig.expansion import transformation

###################################################
# Expansions defined with closed-form polynomials #
###################################################


class hyperbolic_expansion(transformation):
    r"""
    The hyperbolic data expansion function.

    It performs the hyperbolic expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its hyperbolic expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa (\mathbf{x}) = \left[ \cosh(\mathbf{x}), \sinh(\mathbf{x}), \tanh(\mathbf{x}) \right] \in {R}^D,
    \end{equation}
    $$
    where the output dimensions will be $D = 3 m$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the hyperbolic expansion function.

    Attributes
    ----------
    name: str, default = 'hyperbolic_expansion'
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

    def __init__(self, name='hyperbolic_expansion', *args, **kwargs):
        r"""
        The initialization method of the hyperbolic expansion function.

        It initializes a hyperbolic expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'hyperbolic_expansion'
            The name of the hyperbolic expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the hyperbolic expansion function, the expansion space dimension will be
        $$ D = 3m. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m * 3

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the hyperbolic data expansion of the input data and returns the expansion result as
        $$
            \begin{equation}
                \kappa (\mathbf{x}) = \left[ \cosh(\mathbf{x}), \sinh(\mathbf{x}), \tanh(\mathbf{x}) \right] \in {R}^D.
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
        sinh = torch.sinh(x)
        cosh = torch.cosh(x)
        tanh = torch.tanh(x)
        expansion = torch.cat((sinh, cosh, tanh), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class arc_hyperbolic_expansion(transformation):
    r"""
    The arc hyperbolic data expansion function.

    It performs the arc hyperbolic expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its arc hyperbolic expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa (\mathbf{x}) = \left[ \text{arccosh}(\mathbf{x}), \text{arcsinh}(\mathbf{x}), \text{arctanh}(\mathbf{x}) \right] \in {R}^D,
    \end{equation}
    $$
    where the output dimensions will be $D = 3 m$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the arc hyperbolic expansion function.

    Attributes
    ----------
    name: str, default = 'arc_hyperbolic_expansion'
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
    def __init__(self, name='arc_hyperbolic_expansion', *args, **kwargs):
        r"""
        The initialization method of the arc hyperbolic expansion function.

        It initializes a arc hyperbolic expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'arc_hyperbolic_expansion'
            The name of the arc hyperbolic expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the arc hyperbolic expansion function, the expansion space dimension will be
        $$ D = 3m. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m * 3

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the arc hyperbolic data expansion of the input data and returns the expansion result as
        $$
            \begin{equation}
                \kappa (\mathbf{x}) = \left[ \text{arccosh}(\mathbf{x}), \text{arcsinh}(\mathbf{x}), \text{arctanh}(\mathbf{x}) \right] \in {R}^D.
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
        # pre-normalize the input to range [0, 1]
        x = torch.nn.functional.sigmoid(x)
        arcsinh = torch.arcsinh(x)
        arccosh = torch.arccosh(x+1.01)
        arctanh = torch.arctanh(0.99*x)
        expansion = torch.cat((arcsinh, arccosh, arctanh), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class trigonometric_expansion(transformation):
    r"""
    The trigonometric data expansion function.

    It performs the trigonometric expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its trigonometric expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa (\mathbf{x}) = \left[ \cos(\mathbf{x}), \sin(\mathbf{x}), \tan(\mathbf{x}) \right] \in {R}^D,
    \end{equation}
    $$
    where the output dimensions will be $D = 3 m$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the trigonometric expansion function.

    Attributes
    ----------
    name: str, default = 'trigonometric_expansion'
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

    def __init__(self, name='trigonometric_expansion', *args, **kwargs):
        r"""
        The initialization method of the trigonometric expansion function.

        It initializes a trigonometric expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'trigonometric_expansion'
            The name of the trigonometric expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the trigonometric expansion function, the expansion space dimension will be
        $$ D = 3m. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m * 3

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the trigonometric data expansion of the input data and returns the expansion result as
        $$
            \begin{equation}
                \kappa (\mathbf{x}) = \left[ \cos(\mathbf{x}), \sin(\mathbf{x}), \tan(\mathbf{x}) \right] \in {R}^D.
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
        sin = torch.sin(x)
        cos = torch.cos(x)
        tan = torch.tan(x)
        expansion = torch.cat((sin, cos, tan), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class arc_trigonometric_expansion(transformation):
    r"""
    The arc trigonometric data expansion function.

    It performs the arc trigonometric expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its arc trigonometric expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa (\mathbf{x}) = \left[ \arccos(\mathbf{x}), \arcsin(\mathbf{x}), \arctan(\mathbf{x}) \right] \in {R}^D,
    \end{equation}
    $$
    where the output dimensions will be $D = 3 m$.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the arc trigonometric expansion function.

    Attributes
    ----------
    name: str, default = 'arc_trigonometric_expansion'
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
    def __init__(self, name='arc_trigonometric_expansion', *args, **kwargs):
        r"""
        The initialization method of the arc trigonometric expansion function.

        It initializes a arc trigonometric expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'arc_trigonometric_expansion'
            The name of the arc trigonometric expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the arc trigonometric expansion function, the expansion space dimension will be
        $$ D = 3m. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        return m * 3

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the data expansion function.

        It performs the arc trigonometric data expansion of the input data and returns the expansion result as
        $$
            \begin{equation}
                \kappa (\mathbf{x}) = \left[ \arccos(\mathbf{x}), \arcsin(\mathbf{x}), \arctan(\mathbf{x}) \right] \in {R}^D.
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
        # pre-normalize the input to range [0, 1]
        x = torch.nn.functional.sigmoid(x)
        arcsin = torch.arcsin(0.99*x)
        arccos = torch.arccos(0.99*x)
        arctan = torch.arctan(x)
        expansion = torch.cat((arcsin, arccos, arctan), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)