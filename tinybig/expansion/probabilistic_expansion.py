# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
naive probabilistic data expansion functions.

This module contains the naive probabilistic data expansion functions,
including naive_normal_expansion, naive_cauchy_expansion, naive_chi2_expansion,
naive_gamma_expansion, naive_exponential_expansion, and naive_laplace_expansion.
"""

import torch
import torch.nn.functional as F

from tinybig.expansion import transformation

############################################################
# Expansions defined with pure probabilistic distributions #
############################################################


class naive_normal_expansion(transformation):
    r"""
    The naive normal data expansion function.

    It performs the naive normal probabilistic expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its naive normal probabilistic expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
    \end{equation}
    $$
    where $P\left({{x}} | \theta_d\right)$ denotes the probability density function of the normal distribution with hyper-parameter $\theta_d$,
    $$
        \begin{equation}
            P\left(x | \theta_d\right) = P(x| \mu, \sigma)  = \frac{1}{\sigma \sqrt{2 \pi}}\exp^{-\frac{1}{2} (\frac{x-\mu}{\sigma})^2}.
        \end{equation}
    $$

    For naive normal probabilistic expansion, its output expansion dimensions will be $D = md$,
    where $d$ denotes the number of provided distribution hyper-parameters.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the gaussian rbf expansion function.

    Attributes
    ----------
    name: str, default = 'naive_normal_expansion'
        The name of the naive normal expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='naive_normal_expansion', *args, **kwargs):
        r"""
        The initialization method of the naive normal probabilistic expansion function.

        It initializes a naive normal probabilistic expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'naive_normal_expansion'
            The name of the naive normal expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the naive normal probabilistic expansion function, the expansion space dimension will be
        $$ D = m d, $$
        where $d$ denotes the number of provided distribution hyper-parameters.

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
        The forward method of the naive normal probabilistic expansion function.

        It performs the naive normal probabilistic expansion of the input data and returns the expansion result as
        $$
        \begin{equation}
            \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D.
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
        x = x.to('cpu')

        normal_dist_1 = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        normal_x_1 = normal_dist_1.log_prob(x)

        expansion = normal_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class normal_expansion(transformation):
    def __init__(self, name='normal_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        normal_dist_1 = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        normal_x_1 = normal_dist_1.log_prob(x)

        normal_dist_2 = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([1.0]))
        normal_x_2 = normal_dist_2.log_prob(x)

        normal_dist_3 = torch.distributions.normal.Normal(torch.tensor([-1.0]), torch.tensor([1.0]))
        normal_x_3 = normal_dist_3.log_prob(x)

        normal_dist_4 = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
        normal_x_4 = normal_dist_4.log_prob(x)

        normal_dist_5 = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([0.1]))
        normal_x_5 = normal_dist_5.log_prob(x)

        normal_dist_6 = torch.distributions.normal.Normal(torch.tensor([-1.0]), torch.tensor([0.1]))
        normal_x_6 = normal_dist_6.log_prob(x)

        expansion = torch.cat((normal_x_1, normal_x_2, normal_x_3, normal_x_4, normal_x_5, normal_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_cauchy_expansion(transformation):
    r"""
    The naive cauchy data expansion function.

    It performs the naive cauchy probabilistic expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its naive cauchy probabilistic expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
    \end{equation}
    $$
    where $P\left({{x}} | \theta_d\right)$ denotes the probability density function of the cauchy distribution with hyper-parameter $\theta_d$,
    $$
        \begin{equation}
            P\left(x | \theta_d\right) = P(x | x\_0, \gamma) = \frac{1}{\pi \gamma \left[1 +\left( \frac{x-x\_0}{\gamma} \right)^2 \right]}.
        \end{equation}
    $$

    For naive cauchy probabilistic expansion, its output expansion dimensions will be $D = md$,
    where $d$ denotes the number of provided distribution hyper-parameters.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the gaussian rbf expansion function.

    Attributes
    ----------
    name: str, default = 'naive_cauchy_expansion'
        Name of the naive cauchy expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='naive_cauchy_expansion', *args, **kwargs):
        r"""
        The initialization method of the naive cauchy probabilistic expansion function.

        It initializes a naive cauchy probabilistic expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'naive_cauchy_expansion'
            The name of the naive cauchy expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the naive cauchy probabilistic expansion function, the expansion space dimension will be
        $$ D = m d, $$
        where $d$ denotes the number of provided distribution hyper-parameters.

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
        The forward method of the naive cauchy probabilistic expansion function.

        It performs the naive cauchy probabilistic expansion of the input data and returns the expansion result as
        $$
        \begin{equation}
            \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
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
        x = x.to('cpu')

        cauchy_dist_1 = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        cauchy_x_1 = cauchy_dist_1.log_prob(x)

        expansion = cauchy_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class cauchy_expansion(transformation):
    def __init__(self, name='cauchy_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        cauchy_dist_1 = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        cauchy_x_1 = cauchy_dist_1.log_prob(x)

        cauchy_dist_2 = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]), torch.tensor([0.5]))
        cauchy_x_2 = cauchy_dist_2.log_prob(x)

        cauchy_dist_3 = torch.distributions.cauchy.Cauchy(torch.tensor([1.0]), torch.tensor([1.0]))
        cauchy_x_3 = cauchy_dist_3.log_prob(x)

        cauchy_dist_4 = torch.distributions.cauchy.Cauchy(torch.tensor([1.0]), torch.tensor([0.5]))
        cauchy_x_4 = cauchy_dist_4.log_prob(x)

        cauchy_dist_5 = torch.distributions.cauchy.Cauchy(torch.tensor([-1.0]), torch.tensor([1.0]))
        cauchy_x_5 = cauchy_dist_5.log_prob(x)

        cauchy_dist_6 = torch.distributions.cauchy.Cauchy(torch.tensor([-1.0]), torch.tensor([0.5]))
        cauchy_x_6 = cauchy_dist_6.log_prob(x)

        expansion = torch.cat((cauchy_x_1, cauchy_x_2, cauchy_x_3, cauchy_x_4, cauchy_x_5, cauchy_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_chi2_expansion(transformation):
    r"""
    The naive chi2 data expansion function.

    It performs the naive chi2 probabilistic expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its naive chi2 probabilistic expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
    \end{equation}
    $$
    where $P\left({{x}} | \theta_d\right)$ denotes the probability density function of the chi2 distribution with hyper-parameter $\theta_d$,
    $$
        \begin{equation}
            P\left(x | \theta_d\right) = P(x| k) = \frac{1}{2^{\frac{k}{2}} \Gamma(\frac{k}{2})} x^{(\frac{k}{2}-1)} \exp^{-\frac{x}{2}}.
        \end{equation}
    $$

    For naive chi2 probabilistic expansion, its output expansion dimensions will be $D = md$,
    where $d$ denotes the number of provided distribution hyper-parameters.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the gaussian rbf expansion function.

    Attributes
    ----------
    name: str, default = 'naive_chi2_expansion'
        The name of the naive chi2 expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='naive_chi2_expansion', *args, **kwargs):
        r"""
        The initialization method of the naive chi2 probabilistic expansion function.

        It initializes a naive chi2 probabilistic expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'naive_chi2_expansion'
            The name of the naive chi2 expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the naive chi2 probabilistic expansion function, the expansion space dimension will be
        $$ D = m d, $$
        where $d$ denotes the number of provided distribution hyper-parameters.

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
        The forward method of the naive chi2 probabilistic expansion function.

        It performs the naive chi2 probabilistic expansion of the input data and returns the expansion result as
        $$
        \begin{equation}
            \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
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
        x = x.to('cpu')
        x = F.sigmoid(x)+0.001

        chi2_dist_1 = torch.distributions.chi2.Chi2(df=torch.tensor([1.0]))
        chi2_x_1 = chi2_dist_1.log_prob(x)

        expansion = chi2_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class chi2_expansion(transformation):
    def __init__(self, name='chi2_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 2

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')
        x = F.sigmoid(x)+0.001

        chi2_dist_1 = torch.distributions.chi2.Chi2(df=torch.tensor([1.0]))
        chi2_x_1 = chi2_dist_1.log_prob(x)

        chi2_dist_6 = torch.distributions.chi2.Chi2(df=torch.tensor([2.0]))
        chi2_x_6 = chi2_dist_6.log_prob(x)

        expansion = torch.cat((chi2_x_1, chi2_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_gamma_expansion(transformation):
    r"""
    The naive gamma data expansion function.

    It performs the naive gamma probabilistic expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its naive gamma probabilistic expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
    \end{equation}
    $$
    where $P\left({{x}} | \theta_d\right)$ denotes the probability density function of the gamma distribution with hyper-parameter $\theta_d$,
    $$
        \begin{equation}
            P\left(x | \theta_d\right) = P(x | k, \theta) = \frac{1}{\Gamma(k) \theta^k} x^{k-1} \exp^{- \frac{x}{\theta}}.
        \end{equation}
    $$

    For naive gamma probabilistic expansion, its output expansion dimensions will be $D = md$,
    where $d$ denotes the number of provided distribution hyper-parameters.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the gaussian rbf expansion function.

    Attributes
    ----------
    name: str, default = 'naive_gamma_expansion'
        Name of the naive gamma expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='naive_gamma_expansion', *args, **kwargs):
        r"""
        The initialization method of the naive gamma probabilistic expansion function.

        It initializes a naive gamma probabilistic expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'naive_gamma_expansion'
            The name of the naive gamma expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the naive gamma probabilistic expansion function, the expansion space dimension will be
        $$ D = m d, $$
        where $d$ denotes the number of provided distribution hyper-parameters.

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
        The forward method of the naive gamma probabilistic expansion function.

        It performs the naive gamma probabilistic expansion of the input data and returns the expansion result as
        $$
        \begin{equation}
            \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
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
        x = x.to('cpu')
        x = 0.99 * F.sigmoid(x) + 0.001

        gamma_dist_1 = torch.distributions.gamma.Gamma(torch.tensor([0.5]), torch.tensor([1.0]))
        gamma_x_1 = gamma_dist_1.log_prob(x)

        expansion = gamma_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class gamma_expansion(transformation):
    def __init__(self, name='gamma_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        # pre-normalize the input to range [0, 1]
        x = x.to('cpu')
        x = 0.99 * F.sigmoid(x) + 0.001

        gamma_dist_1 = torch.distributions.gamma.Gamma(torch.tensor([0.5]), torch.tensor([1.0]))
        gamma_x_1 = gamma_dist_1.log_prob(x)

        gamma_dist_2 = torch.distributions.gamma.Gamma(torch.tensor([0.5]), torch.tensor([2.0]))
        gamma_x_2 = gamma_dist_2.log_prob(x)

        gamma_dist_3 = torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        gamma_x_3 = gamma_dist_3.log_prob(x)

        gamma_dist_4 = torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([2.0]))
        gamma_x_4 = gamma_dist_4.log_prob(x)

        gamma_dist_5 = torch.distributions.gamma.Gamma(torch.tensor([2.0]), torch.tensor([1.0]))
        gamma_x_5 = gamma_dist_5.log_prob(x)

        gamma_dist_6 = torch.distributions.gamma.Gamma(torch.tensor([2.0]), torch.tensor([2.0]))
        gamma_x_6 = gamma_dist_6.log_prob(x)

        expansion = torch.cat((gamma_x_1, gamma_x_2, gamma_x_3, gamma_x_4, gamma_x_5, gamma_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_laplace_expansion(transformation):
    r"""
    The naive laplace data expansion function.

    It performs the naive laplace probabilistic expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its naive laplace probabilistic expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
    \end{equation}
    $$
    where $P\left({{x}} | \theta_d\right)$ denotes the probability density function of the laplace distribution with hyper-parameter $\theta_d$,
    $$
        \begin{equation}
            P\left(x | \theta_d\right) = P(x| \mu, b) = \frac{1}{2b} \exp^{\left(- \frac{|x-\mu|}{b} \right)}.
        \end{equation}
    $$

    For naive laplace probabilistic expansion, its output expansion dimensions will be $D = md$,
    where $d$ denotes the number of provided distribution hyper-parameters.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the gaussian rbf expansion function.

    Attributes
    ----------
    name: str, default = 'naive_laplace_expansion'
        Name of the naive laplace expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='naive_laplace_expansion', *args, **kwargs):
        r"""
        The initialization method of the naive laplace probabilistic expansion function.

        It initializes a naive laplace probabilistic expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'naive_laplace_expansion'
            The name of the naive laplace expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the naive laplace probabilistic expansion function, the expansion space dimension will be
        $$ D = m d, $$
        where $d$ denotes the number of provided distribution hyper-parameters.

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
        The forward method of the naive laplace probabilistic expansion function.

        It performs the naive laplace probabilistic expansion of the input data and returns the expansion result as
        $$
        \begin{equation}
            \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
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
        x = x.to('cpu')

        laplace_dist_1 = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        laplace_x_1 = laplace_dist_1.log_prob(x)

        expansion = laplace_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class laplace_expansion(transformation):
    def __init__(self, name='laplace_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        laplace_dist_1 = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        laplace_x_1 = laplace_dist_1.log_prob(x)

        laplace_dist_2 = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([2.0]))
        laplace_x_2 = laplace_dist_2.log_prob(x)

        laplace_dist_3 = torch.distributions.laplace.Laplace(torch.tensor([1.0]), torch.tensor([1.0]))
        laplace_x_3 = laplace_dist_3.log_prob(x)

        laplace_dist_4 = torch.distributions.laplace.Laplace(torch.tensor([1.0]), torch.tensor([2.0]))
        laplace_x_4 = laplace_dist_4.log_prob(x)

        laplace_dist_5 = torch.distributions.laplace.Laplace(torch.tensor([-1.0]), torch.tensor([1.0]))
        laplace_x_5 = laplace_dist_5.log_prob(x)

        laplace_dist_6 = torch.distributions.laplace.Laplace(torch.tensor([-1.0]), torch.tensor([2.0]))
        laplace_x_6 = laplace_dist_6.log_prob(x)

        expansion = torch.cat((laplace_x_1, laplace_x_2, laplace_x_3, laplace_x_4, laplace_x_5, laplace_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_exponential_expansion(transformation):
    r"""
    The naive exponential data expansion function.

    It performs the naive exponential probabilistic expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its naive exponential probabilistic expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
    \end{equation}
    $$
    where $P\left({{x}} | \theta_d\right)$ denotes the probability density function of the exponential distribution with hyper-parameter $\theta_d$,
    $$
        \begin{equation}
            P\left(x | \theta_d\right) = P(x | \lambda) =   \begin{cases}
                                                                \lambda \exp^{- \lambda x} & \text{ for } x \ge 0,\\\\
                                                                0 & \text{ otherwise}.
                                                            \end{cases}
        \end{equation}
    $$

    For naive exponential probabilistic expansion, its output expansion dimensions will be $D = md$,
    where $d$ denotes the number of provided distribution hyper-parameters.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the gaussian rbf expansion function.

    Attributes
    ----------
    name: str, default = 'naive_exponential_expansion'
        Name of the naive exponential expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name='naive_exponential_expansion', *args, **kwargs):
        r"""
        The initialization method of the naive exponential probabilistic expansion function.

        It initializes a naive exponential probabilistic expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'naive_exponential_expansion'
            The name of the naive exponential expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the naive exponential probabilistic expansion function, the expansion space dimension will be
        $$ D = m d, $$
        where $d$ denotes the number of provided distribution hyper-parameters.

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
        The forward method of the naive exponential probabilistic expansion function.

        It performs the naive exponential probabilistic expansion of the input data and returns the expansion result as
        $$
        \begin{equation}
            \kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x}} | \theta\_1\right), \log P\left({\mathbf{x} } | \theta\_2\right), \cdots, \log P\left({\mathbf{x} } | \theta\_d\right)  \right] \in {R}^D
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
        x = x.to('cpu')
        x = 0.99 * F.sigmoid(x) + 0.001

        exponential_dist_1 = torch.distributions.exponential.Exponential(torch.tensor([0.5]))
        exponential_x_1 = exponential_dist_1.log_prob(x)

        expansion = exponential_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class exponential_expansion(transformation):
    def __init__(self, name='exponential_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        # pre-normalize the input to range [0, 1]
        x = x.to('cpu')
        x = 0.99 * F.sigmoid(x) + 0.001

        exponential_dist_1 = torch.distributions.exponential.Exponential(torch.tensor([0.5]))
        exponential_x_1 = exponential_dist_1.log_prob(x)

        exponential_dist_2 = torch.distributions.exponential.Exponential(torch.tensor([0.75]))
        exponential_x_2 = exponential_dist_2.log_prob(x)

        exponential_dist_3 = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
        exponential_x_3 = exponential_dist_3.log_prob(x)

        exponential_dist_4 = torch.distributions.exponential.Exponential(torch.tensor([2.0]))
        exponential_x_4 = exponential_dist_4.log_prob(x)

        exponential_dist_5 = torch.distributions.exponential.Exponential(torch.tensor([3.0]))
        exponential_x_5 = exponential_dist_5.log_prob(x)

        exponential_dist_6 = torch.distributions.exponential.Exponential(torch.tensor([5.0]))
        exponential_x_6 = exponential_dist_6.log_prob(x)

        expansion = torch.cat((exponential_x_1, exponential_x_2, exponential_x_3, exponential_x_4, exponential_x_5, exponential_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class hybrid_probabilistic_expansion(transformation):
    def __init__(self, name='hybrid_probabilistic_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        normal_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        normal_x = normal_dist.log_prob(x)

        cauchy_dist = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        cauchy_x = cauchy_dist.log_prob(x)

        chi_dist = torch.distributions.chi2.Chi2(df=torch.tensor([1.0]))
        chi_x = chi_dist.log_prob(0.99 * F.sigmoid(x) + 0.001)

        gamma_dist = torch.distributions.gamma.Gamma(torch.tensor([0.5]), torch.tensor([1.0]))
        gamma_x = gamma_dist.log_prob(0.99 * F.sigmoid(x) + 0.001)

        laplace_dist = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        laplace_x = laplace_dist.log_prob(x)

        exponential_dist = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
        exponential_x = exponential_dist.log_prob(0.99 * F.sigmoid(x) + 0.001)

        expansion = torch.cat((normal_x, cauchy_x, chi_x, gamma_x, laplace_x, exponential_x), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)
