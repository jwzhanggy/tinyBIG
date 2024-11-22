# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###################################################
# Orthogonal Polynomial based Expansion Functions #
###################################################

"""
The orthogonal polynomial data expansion functions.

This module contains the orthogonal polynomial data expansion functions, including
    hermite_expansion,
    laguerre_expansion,
    legendre_expansion,
    gegenbauer_expansion,
    bessel_expansion,
    reverse_bessel_expansion,
    fibonacci_expansion,
    lucas_expansion,
"""

import torch.nn

from tinybig.expansion import transformation


class hermite_expansion(transformation):
    r"""
        The hermite expansion function.

        Applies Hermite polynomial expansion to input data.

        Notes
        ---------
        Hermite polynomials, first defined by Pierre-Simon Laplace in 1810 and studied in detail by Pafnuty Chebyshev in 1859, were later named after Charles Hermite, who published work on these polynomials in 1864. The Hermite polynomials can be defined in various forms:

        __Probabilist's Hermite polynomials:__

        $$
            \begin{equation}
            He_n(x) = (-1)^n \exp \left(\frac{x^2}{2} \right) \frac{\mathrm{d}^n}{\mathrm{d}x^n} \exp \left(- \frac{x^2}{2} \right).
            \end{equation}
        $$

        __Physicist's Hermite polynomials:__

        $$
            \begin{equation}
            H_n(x) = (-1)^n \exp \left(x^2 \right) \frac{\mathrm{d}^n}{\mathrm{d}x^n} \exp \left(- x^2 \right).
            \end{equation}
        $$

        These two forms are not identical but can be reduced to each via rescaling:

        $$
            \begin{equation}
            H_n(x) = 2^{\frac{n}{2}} He_n (\sqrt{2}x) \text{, and } He_n(x) = 2^{-\frac{n}{2}} H_n \left(\frac{x}{\sqrt{2}} \right).
            \end{equation}
        $$

        In this paper, we will use the Probabilist's Hermite polynomials for to define the data expansion function by default, which can be formally defined as the following recursive representations:

        $$
            \begin{equation}
            He_{n+1}(x)  = x He_n(x) - n He_{n-1}(x), \forall n \ge 1.
            \end{equation}
        $$

        Some examples of the Probabilist's Hermite polynomials are also illustrated as follows:

        $$
            \begin{equation}
            \begin{aligned}
            He_{0}(x) &= 1;\\
            He_{1}(x) &= x;\\
            He_{2}(x) &= x^2 - 1;\\
            He_{3}(x) &= x^3 - 3x;\\
            He_{4}(x) &= x^4 - 6x^2 + 3.\\
            \end{aligned}
            \end{equation}
        $$

        Based on the Probabilist's Hermite polynomials, we can define the data expansion function with order $d$ as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d)  = \left[ He_1(\mathbf{x}), He_2(\mathbf{x}), \cdots, He_d(\mathbf{x}) \right] \in R^D,
            \end{equation}
        $$

        where $d$ is the order hyper-parameter and the output dimension $D = md$.

        Attributes
        ----------
        d : int
            The degree of Hermite polynomial expansion.

        Methods
        -------
        calculate_D(m: int)
            Calculates the output dimension after expansion.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Performs Hermite polynomial expansion on the input tensor.
    """

    def __init__(self, name: str = 'chebyshev_polynomial_expansion', d: int = 2, *args, **kwargs):
        """
            Initializes the Hermite expansion.

            Parameters
            ----------
            name : str, optional
                Name of the expansion. Defaults to 'chebyshev_polynomial_expansion'.
            d : int, optional
                Degree of Hermite polynomial expansion. Defaults to 2.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        """
            Calculates the output dimension after Hermite polynomial expansion.

            Parameters
            ----------
            m : int
                Input dimension.

            Returns
            -------
            int
                Output dimension.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Performs Hermite polynomial expansion on the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, input_dim)`.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Expanded tensor.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected dimensions.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = x * expansion[:, :, n-1].clone() - (n-1) * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class laguerre_expansion(transformation):
    r"""
        The laguerre expansion function.

        Applies Laguerre polynomial expansion to input data.

        Notes
        ----------

        In mathematics, the Laguerre polynomials, named after Edmond Laguerre, are the nontrivial solutions of Laguerre's differential equation:

        $$
            \begin{equation}
            x y'' + (\alpha + 1 -x) y' + d y = 0,
            \end{equation}
        $$

        where $y = y(x)$ is a function of variable $x$. Notations $y'$ and $y''$ denote first- and second-order derivatives of function $y$ with respect to variable $x$. Term $d \in N$ is a non-negative integer and $\alpha \in R$ is a hyper-parameter.

        The closed-form of the Laguerre polynomials can be represented as follows:

        $$
            \begin{equation}
            P^{(\alpha)}_n(x) = \frac{e^x}{n!} \frac{\mathrm{d}^n}{\mathrm{d} x^n} (e^{-x} x^n) = \frac{x^{-\alpha}}{n!} \left( \frac{\mathrm{d}}{\mathrm{d} x} - 1 \right)^n x^{n + \alpha},
            \end{equation}
        $$

        where $\frac{\mathrm{d}}{\mathrm{d} x}$ denotes the derivative operator.

        In practice, the Laguerre polynomials can be recursively defined as follows, which will be used for defining the data expansion function below. Specifically, when $\alpha = 0$, the above Laguerre polynomials are also known as simple Laguerre polynomials.

        __Base cases $n=0$ and $n=1$:__

        $$
            \begin{equation}
            P^{(\alpha)}_0(x) = 1 \text{, and } P^{(\alpha)}_1(x) = 1 + \alpha -  x.
            \end{equation}
        $$

        __High-order cases with degree $n \ge 2$:__

        $$
            \begin{equation}
            P^{(\alpha)}_n(x) = \frac{(2n-1+\alpha-x) P^{(\alpha)}_{n-1}(x) - (n-1+\alpha) P^{(\alpha)}_{n-2}(x) }{n}
            \end{equation}
        $$

        The recursive-form representations of the Laguerre polynomials can be used to define the data expansion function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d, \alpha) = \left[ P^{(\alpha)}_1(\mathbf{x}), P^{(\alpha)}_2(\mathbf{x}), \cdots, P^{(\alpha)}_d(\mathbf{x}) \right] \in R^D,
            \end{equation}
        $$

        where $d$ and $\alpha$ are the function hyper-parameters and the output dimension $D = md$.

        Attributes
        ----------
        d : int
            The degree of Laguerre polynomial expansion.
        alpha : float
            Parameter controlling the Laguerre polynomial.

        Methods
        -------
        calculate_D(m: int)
            Calculates the output dimension after expansion.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Performs Laguerre polynomial expansion on the input tensor.
    """
    def __init__(self, name='laguerre_polynomial_expansion', d: int = 2, alpha: float = 1.0, *args, **kwargs):
        """
            Initializes the Laguerre polynomial expansion transformation.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'laguerre_polynomial_expansion'.
            d : int, optional
                The maximum order of Laguerre polynomials for expansion. Defaults to 2.
            alpha : float, optional
                The alpha parameter for generalized Laguerre polynomials. Defaults to 1.0.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d
        self.alpha = alpha

    def calculate_D(self, m: int):
        """
            Calculates the output dimension after Laguerre polynomial expansion.

            Parameters
            ----------
            m : int
                Input dimension.

            Returns
            -------
            int
                Output dimension after expansion.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Performs Laguerre polynomial expansion on the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, input_dim)`.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Expanded tensor of shape `(batch_size, expanded_dim)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected dimensions.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x + 1.0 + self.alpha
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (2*n-1+self.alpha-x)/n * expansion[:, :, n-1].clone() - (n-1+self.alpha)/n * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class legendre_expansion(transformation):
    r"""
        The legendre expansion function.

        Applies Legendre polynomial expansion to input data.

        Notes
        ----------
        The Legendre polynomials, named after mathematician Adrien-Marie Legendre, are defined as an orthogonal system over the interval $[-1, 1]$, where the polynomial term $P_n(x)$ of degree $n$ satisfies the following equation:

        $$
            \begin{equation}
            \int_{-1}^{+1} P_m(x) P_n(x) dx = 0, \text{ if } m \neq n.
            \end{equation}
        $$

        Specifically, according to Bonnet's formula, the Legendre polynomials can be recursively represented as follows:

        __Base cases $n=0$ and $n=1$:__

        $$
            \begin{equation}
            P_0(x) = 1 \text{, and } P_1(x) = x.
            \end{equation}
        $$

        __High-order cases with degree $n \ge 2$:__

        $$
            \begin{equation}
            P_n(x) = \frac{x(2n-1) P_{n-1}(x) - (n-1) P_{n-2}(x) }{n}
            \end{equation}
        $$

        The Legendre polynomials help define the data expansion function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d) = \left[ P_1(\mathbf{x}), P_2(\mathbf{x}), \cdots, P_d(\mathbf{x}) \right] \in R^D,
            \end{equation}
        $$

        where the output dimension $D = md$.

        Attributes
        ----------
        d : int
            The degree of Legendre polynomial expansion.

        Methods
        -------
        calculate_D(m: int)
            Calculates the output dimension after expansion.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Performs Legendre polynomial expansion on the input tensor.
    """
    def __init__(self, name='legendre_polynomial_expansion', d: int = 2, *args, **kwargs):
        """
            Initializes the Legendre polynomial expansion transformation.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'legendre_polynomial_expansion'.
            d : int, optional
                The maximum order of Legendre polynomials for expansion. Defaults to 2.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        """
            Calculates the output dimension after Legendre polynomial expansion.

            Parameters
            ----------
            m : int
                Input dimension.

            Returns
            -------
            int
                Output dimension after expansion.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Performs Legendre polynomial expansion on the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, input_dim)`.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Expanded tensor of shape `(batch_size, expanded_dim)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected dimensions.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (2*n-1)/n * x * expansion[:, :, n-1].clone() - (n-1)/n * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class gegenbauer_expansion(transformation):
    r"""
        The gegenbauer expansion function.

        Applies Gegenbauer polynomial expansion to input data.

        Notes
        ----------

        The Gegenbauer polynomials, named after mathematician Leopold Gegenbauer, are orthogonal polynomials that generalize both the Legendre and Chebyshev polynomials, and are special cases of Jacobi polynomials.

        Formally, the Gegenbauer polynomials are particular solutions of the Gegenbauer differential equation:

        $$
            \begin{equation}
            (1 - x^2) y'' - (2 \alpha + 1) x y' + d(d+2 \alpha) y = 0,
            \end{equation}
        $$

        where $y = y(x)$ is a function of variable $x$ and $d \in N$ is a non-negative integer.

        When $\alpha = \frac{1}{2}$, the Gegenbauer polynomials reduce to the Legendre polynomials introduced earlier; when $\alpha = 1$, they reduce to the Chebyshev polynomials of the second kind.

        The Gegenbauer polynomials can be recursively defined as follows:

        __Base cases $n=0$ and $n=1$:__

        $$
            \begin{equation}
            P^{(\alpha)}_0(x) = 1 \text{, and } P^{(\alpha)}_1(x) = 2 \alpha x.
            \end{equation}
        $$

        __High-order cases with degree $n \ge 2$:__

        $$
            \begin{equation}
            P^{(\alpha)}_n(x) = \frac{2x(n-1+\alpha) P^{(\alpha)}_{n-1}(x) - (n+2\alpha -2) P^{(\alpha)}_{n-2}(x) }{n}
            \end{equation}
        $$

        Based on the Gegenbauer polynomials, we can define the expansion function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d, \alpha) = \left[ P^{(\alpha)}_1(\mathbf{x}), P^{(\alpha)}_2(\mathbf{x}), \cdots, P^{(\alpha)}_d(\mathbf{x}) \right] \in R^D,
            \end{equation}
        $$

        where the output dimension $D = md$.


        Attributes
        ----------
        d : int
            The degree of Gegenbauer polynomial expansion.
        alpha : float
            Parameter controlling the Gegenbauer polynomial.

        Methods
        -------
        calculate_D(m: int)
            Calculates the output dimension after expansion.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Performs Gegenbauer polynomial expansion on the input tensor.
    """
    def __init__(self, name='gegenbauer_polynomial_expansion', d: int = 2, alpha: float = 1.0, *args, **kwargs):
        """
            Initializes the Gegenbauer polynomial expansion transformation.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'gegenbauer_polynomial_expansion'.
            d : int, optional
                The maximum order of Gegenbauer polynomials for expansion. Defaults to 2.
            alpha : float, optional
                The alpha parameter for Gegenbauer polynomials. Defaults to 1.0.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d
        self.alpha = alpha

    def calculate_D(self, m: int):
        """
            Calculates the output dimension after Gegenbauer polynomial expansion.

            Parameters
            ----------
            m : int
                Input dimension.

            Returns
            -------
            int
                Output dimension after expansion.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Performs Gegenbauer polynomial expansion on the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, input_dim)`.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Expanded tensor of shape `(batch_size, expanded_dim)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected dimensions.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = 2 * self.alpha * x
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (n-1+self.alpha)/n * 2*x * expansion[:, :, n-1].clone() - (n-2+2*self.alpha)/n * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class bessel_expansion(transformation):
    r"""
        The bessel expansion function.

        Applies Bessel polynomial expansion to input data.

        Notes
        ----------

        Formally, the Bessel polynomials are an orthogonal sequence of polynomials with the following closed-form representation:

        $$
            \begin{equation}
            B_n(x) = \sum_{k=0}^n \frac{(n+k)!}{(n-k)! k!} \left( \frac{x}{2}\right)^k.
            \end{equation}
        $$

        The Bessel polynomials can be recursively defined as follows:

        __Base cases $n=0$ and $n=1$:__

        $$
            \begin{equation}
            B_0(x) = 1 \text{, and } B_1(x) = x + 1.
            \end{equation}
        $$

        __High-order cases with degree $n \ge 2$:__

        $$
            \begin{equation}
            B_n(x) = (2n - 1) x B_{n-1}(x) + B_{n-2}(x).
            \end{equation}
        $$

        The Bessel polynomials can be used to define the data expansion functions as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d) = \left[ B_1(\mathbf{x}), B_2(\mathbf{x}), \cdots, B_d(\mathbf{x}) \right] \in R^D,
            \end{equation}
        $$

        where the output dimension $D = md$.

        Attributes
        ----------
        d : int
            The degree of Bessel polynomial expansion.

        Methods
        -------
        calculate_D(m: int)
            Calculates the output dimension after expansion.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Performs Bessel polynomial expansion on the input tensor.
    """
    def __init__(self, name='bessel_polynomial_expansion', d: int = 2, *args, **kwargs):
        """
            Initializes the Bessel polynomial expansion transformation.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'bessel_polynomial_expansion'.
            d : int, optional
                The maximum order of Bessel polynomials for expansion. Defaults to 2.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        """
            Calculates the output dimension after Bessel polynomial expansion.

            Parameters
            ----------
            m : int
                Input dimension.

            Returns
            -------
            int
                Output dimension after expansion.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Performs Bessel polynomial expansion on the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, input_dim)`.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Expanded tensor of shape `(batch_size, expanded_dim)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected dimensions.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x + 1
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (2*n-1) * x * expansion[:, :, n-1].clone() + expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class reverse_bessel_expansion(transformation):
    r"""
        The reverse bessel expansion function.

        Applies reverse Bessel polynomial expansion to input data.

        Notes
        ----------

        Formally, the reverse Bessel polynomials are an orthogonal sequence of polynomials with the following closed-form representation:

        $$
            \begin{equation}
            R_n(x) = x^n B_n \left( \frac{1}{x} \right) = \sum_{k=0}^n \frac{(n+k)!}{(n-k)! k!} \frac{x^{n-k}}{2^k} .
            \end{equation}
        $$

        The reverse Bessel polynomials can be recursively defined as follows:

        __Base cases $n=0$ and $n=1$:__

        $$
            \begin{equation}
            R_0(x) = 1 \text{, and } R_1(x) = x + 1.
            \end{equation}
        $$

        __High-order cases with degree $n \ge 2$:__

        $$
            \begin{equation}
            R_n(x) = (2n - 1) R_{n-1}(x) + x^2 R_{n-2}(x)
            \end{equation}
        $$

        The reverse Bessel polynomials can be used to define the data expansion functions as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d) =  \left[ R_1(\mathbf{x}), R_2(\mathbf{x}), \cdots, R_d(\mathbf{x}) \right] \in R^D,
            \end{equation}
        $$

        where the output dimension $D = md$.

        Attributes
        ----------
        d : int
            The degree of reverse Bessel polynomial expansion.

        Methods
        -------
        calculate_D(m: int)
            Calculates the output dimension after expansion.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Performs reverse Bessel polynomial expansion on the input tensor.
    """
    def __init__(self, name='reverse_bessel_polynomial_expansion', d: int = 2, *args, **kwargs):
        """
            Initializes the reverse Bessel polynomial expansion transformation.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'reverse_bessel_polynomial_expansion'.
            d : int, optional
                The maximum order of reverse Bessel polynomials for expansion. Defaults to 2.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        """
            Calculates the output dimension after reverse Bessel polynomial expansion.

            Parameters
            ----------
            m : int
                Input dimension.

            Returns
            -------
            int
                Output dimension after expansion.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Performs reverse Bessel polynomial expansion on the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, input_dim)`.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Expanded tensor of shape `(batch_size, expanded_dim)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected dimensions.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x + 1
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (2*n-1) * expansion[:, :, n-1].clone() + x * x * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class fibonacci_expansion(transformation):
    r"""
        The fibonacci expansion function.

        Applies Fibonacci polynomial expansion to input data.

        Notes
        ---------

        Formally, the Fibonacci polynomials are a polynomial sequence that can be considered a generalization of the Fibonacci numbers,
        which can be recursively represented as follows:

        __Base cases $n=0$ and $n=1$:__

        $$
            \begin{equation}
            F_0(x) = 0 \text{, and } F_1(x) = 1.
            \end{equation}
        $$

        __High-order cases with degree $n \ge 2$:__

        $$
            \begin{equation}
            F_n(x) = x F_{n-1}(x) + F_{n-2}(x).
            \end{equation}
        $$

        Based on these recursive representations, we can illustrate some examples of the Fibonacci polynomials as follows:

        $$
            \begin{equation}
            \begin{aligned}
            F_0(x) &= 0 \\
            F_1(x) &= 1 \\
            F_2(x) &= x \\
            F_3(x) &= x^2 + 1 \\
            F_4(x) &= x^3 + 2x \\
            F_5(x) &= x^4 + 3x^2 + 1 \\
            \end{aligned}
            \end{equation}
        $$

        Based on the above Fibonacci polynomials, we can define the data expansion functions as follows:
        $$
            \begin{equation}
            \kappa(\mathbf{x} | d) = \left[ F_1(\mathbf{x}), F_2(\mathbf{x}), \cdots, F_d(\mathbf{x}) \right] \in R^D,
            \end{equation}
        $$
        where the output dimension $D = md$.

        Attributes
        ----------
        d : int
            The degree of Fibonacci polynomial expansion.

        Methods
        -------
        calculate_D(m: int)
            Calculates the output dimension after expansion.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Performs Fibonacci polynomial expansion on the input tensor.
    """
    def __init__(self, name='fibonacci_polynomial_expansion', d: int = 2, *args, **kwargs):
        """
            Initializes the Fibonacci polynomial expansion transformation.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'fibonacci_polynomial_expansion'.
            d : int, optional
                The maximum order of Fibonacci polynomials for expansion. Defaults to 2.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        """
            Calculates the output dimension after Fibonacci polynomial expansion.

            Parameters
            ----------
            m : int
                Input dimension.

            Returns
            -------
            int
                Output dimension after expansion.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Performs Fibonacci polynomial expansion on the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, input_dim)`.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Expanded tensor of shape `(batch_size, expanded_dim)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected dimensions.
        """

        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.zeros(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = torch.ones(size=[x.size(0), x.size(1)]).to(device)
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = x * expansion[:, :, n-1].clone() + expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class lucas_expansion(transformation):
    r"""
        The lucas expansion function.

        Applies Lucas polynomial expansion to input data.

        Notes
        ---------

        Formally, the Lucas polynomials are generated from the Lucas numbers in an analogous manner.
        The Lucas polynomials can be viewed as identical to the Fibonacci polynomials but with different base case representations,
        which can be recursively defined as follows:

        __Base cases $n=0$ and $n=1$:__

        $$
            \begin{equation}
            L_0(x) = 2 \text{, and } L_1(x) = x.
            \end{equation}
        $$

        __High-order cases with degree $n \ge 2$:__

        $$
            \begin{equation}
            L_n(x) = x L_{n-1}(x) + L_{n-2}(x).
            \end{equation}
        $$

        Based on these recursive representations, we can illustrate some examples of the Lucas polynomials as follows:

        $$
            \begin{equation}
            \begin{aligned}
            L_0(x) &= 2 \\
            L_1(x) &= x \\
            L_2(x) &= x^2 + 2 \\
            L_3(x) &= x^3 + 3x \\
            L_4(x) &= x^4 + 4x^2 + 2 \\
            L_5(x) &= x^5 + 5x^3 + 5x \\
            \end{aligned}
            \end{equation}
        $$

        Based on the above Lucas polynomials, we can define the data expansion functions as follows:
        $$
            \begin{equation}
            \kappa(\mathbf{x} | d) = \left[ L_1(\mathbf{x}), L_2(\mathbf{x}), \cdots, L_d(\mathbf{x}) \right] \in R^D,
            \end{equation}
        $$
        where the output dimension $D = md$.

        Attributes
        ----------
        d : int
            The degree of Lucas polynomial expansion.

        Methods
        -------
        calculate_D(m: int)
            Calculates the output dimension after expansion.
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Performs Lucas polynomial expansion on the input tensor.
    """
    def __init__(self, name='lucas_polynomial_expansion', d: int = 2, *args, **kwargs):
        """
            Initializes the Lucas polynomial expansion transformation.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'lucas_polynomial_expansion'.
            d : int, optional
                The maximum order of Lucas polynomials for expansion. Defaults to 2.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        """
            Calculates the output dimension after Lucas polynomial expansion.

            Parameters
            ----------
            m : int
                Input dimension.

            Returns
            -------
            int
                Output dimension after expansion.
        """
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Performs Lucas polynomial expansion on the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, input_dim)`.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Expanded tensor of shape `(batch_size, expanded_dim)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected dimensions.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = 2 * torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = x * expansion[:, :, n-1].clone() + expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)

