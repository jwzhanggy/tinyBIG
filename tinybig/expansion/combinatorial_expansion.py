# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
combinatorial data expansion functions.

This module contains the combinatorial data expansion functions,
including combinatorial_expansion, and combinatorial_normal_expansion.
"""

import torch
from scipy.special import comb

from tinybig.expansion import transformation


############################
# Combinatorial Expansions #
############################

class combinatorial_expansion(transformation):
    r"""
    The combinatorial data expansion function.

    It performs the combinatorial data expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its combinatorial data expansion can be represented as follows:
    $$
    \begin{equation}
        \kappa(\mathbf{x}) = \left[ {\mathbf{x} \choose 1}, {\mathbf{x} \choose 2}, \cdots, {\mathbf{x} \choose d} \right] \in {R}^D.
    \end{equation}
    $$

    Formally, given a data instance featured by a variable set $\mathcal{X} = \{X_1, X_2, \cdots, X_m\}$
    (here, we use the upper-case $X_i$ to denote the variable of the $i_{th}$ feature),
    we can represent the possible combinations of $d$ terms selected from $\mathcal{X}$ as follows:
    $$
        \begin{equation}
            {\mathcal{X} \choose d} = \{ \mathcal{C} | \mathcal{C} \subset \mathcal{X} \land |\mathcal{C}| = d \},
        \end{equation}
    $$
    where $\mathcal{C}$ denotes a subset of $\mathcal{X}$ containing no duplicated elements and
    the size of the output set ${\mathcal{X} \choose d}$ will be equal to ${m \choose d}$.

    Some simple examples with $d=1$, $d=2$ and $d=3$ are illustrated as follows:
    $$
        \begin{align}
            d = 1:\ \  &{\mathcal{X} \choose 1} = \\{\\{X_i\\} | X_i \in \mathcal{X} \\},\\\\
            d = 2:\ \  &{\mathcal{X} \choose 2} = \\{\\{X_i, X_j\\} | X_i, X_j \in \mathcal{X} \land X_i \neq X_j \\},\\\\
            d = 3:\ \  &{\mathcal{X} \choose 3} = \\{\\{X_i, X_j, X_k\\} | X_i, X_j, X_k \in \mathcal{X} \land X_i \neq X_j \land X_i \neq X_k \land X_j \neq X_k \\}.
        \end{align}
    $$

    For combinatorial data expansion, its output expansion dimensions will be $D = \sum_{i=1}^d i \cdot {m \choose i}$,
    where $d$ denotes the combinatorial expansion order parameter.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the gaussian rbf expansion function.

    Attributes
    ----------
    name: str, default = 'combinatorial_expansion'
        The name of the combinatorial expansion function.
    d: int, default = 2
        The combinatorial expansion order.
    with_replacement: bool, default = False
        The with_replacement tag for the random combination.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name: str = 'combinatorial_expansion', d: int = 2, with_replacement: bool = False, *args, **kwargs):
        r"""
        The initialization method of the combinatorial expansion function.

        It initializes a combinatorial expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'combinatorial_expansion'
            The name of the combinatorial expansion function.
        d: int, default = 2
            The order of random combinations.
        with_replacement: bool, default = False
            The replacement boolean tag.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d
        self.with_replacement = with_replacement

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the combinatorial expansion function, the expansion space dimension will be
        $$ D = \sum_{i=1}^d i \cdot {m \choose i}. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        assert type(self.d) is int and self.d >= 1
        return int(sum([r*comb(m, r) for r in range(1, self.d+1)]))

    @staticmethod
    def random_combinations(x: torch.Tensor, r: int = 2, with_replacement: bool = False):
        """
        The random combination generation method.

        It generates the random combinations of $r$ elements in the input data vector, where $r$ is a provided parameter.
        The method will call the torch.combinations method.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        r: int, default = 2
            The number of elements to be combined.
        with_replacement: bool, default = False
            The replacement boolean tag.

        Returns
        -------
        torch.Tensor
            The tensor including all the combinations of elements of size r.
        """
        assert 0 <= r <= x.numel()
        return torch.combinations(input=x, r=r, with_replacement=with_replacement)

    @staticmethod
    def combinatorial(x: torch.Tensor, d: int = 2, device='cpu', with_replacement: bool = False, *args, **kwargs):
        """
        The combinatorial generation method.

        It generates the random combinations of elements from the input data vector.
        The number of combined elements ranges from 1 to the provided order parameter d.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        d: int, default = 2
            The order of the random combinations.
        device: str, default = 'cpu'
            The device to perform and host the combinations.
        with_replacement: bool, default = False
            The replacement boolean tag.
        Returns
        -------
        torch.Tensor
            The tensor including all the combinations of elements of sizes from 1 to d.
        """
        # x: [batch, m]
        if len(x.shape) == 2:
            expansion = []
            for r in range(1, d + 1):
                degree_expansion = []
                for i in range(x.size(0)):
                    degree_expansion.append(combinatorial_expansion.random_combinations(x=x[i,:], r=r, with_replacement=with_replacement))
                expansion.append(degree_expansion)
            # expansion: [0:d-1, 0:batch-1, 0:(m choose d)-1]
            return expansion
        elif len(x.shape) == 1:
            # x: [m]
            expansion = []
            for r in range(1, d + 1):
                expansion.append(combinatorial_expansion.random_combinations(x=x, r=r, with_replacement=with_replacement))
            # expansion: [0:d-1, 0:(m choose d)-1]
            return expansion
        else:
            raise ValueError("Input x can only be 2d or 1d, higher dimensional inputs are not supported yet...")

    def forward(self, x: torch.Tensor, device='cpu', with_replacement: bool = False, *args, **kwargs):
        r"""
        The forward method of the combinatorial expansion function.

        It performs the combinatorial data expansion of the input data and returns the expansion result as
        $$
        \begin{equation}
            \kappa(\mathbf{x}) = \left[ {\mathbf{x} \choose 1}, {\mathbf{x} \choose 2}, \cdots, {\mathbf{x} \choose d} \right] \in {R}^D.
        \end{equation}
        $$


        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.
        with_replacement: bool, default = False
            The replacement boolean tag.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        with_replacement = with_replacement if with_replacement is not None else self.with_replacement
        return self.combinatorial(x=x, d=self.d, device=device, with_replacement=with_replacement, *args, **kwargs)


class combinatorial_normal_expansion(combinatorial_expansion):
    r"""
    The combinatorial normal data expansion function.

    It performs the combinatorial normal probabilistic expansion of the input vector, and returns the expansion result.
    The class inherits from the base expansion class (i.e., the transformation class in the module directory).

    ...

    Notes
    ----------
    For input vector $\mathbf{x} \in R^m$, its combinatorial normal probabilistic expansion can be represented as follows:
    $$
    \begin{equation}
        kappa(\mathbf{x} | \boldsymbol{\theta}) = \left[ \log P\left({\mathbf{x} \choose 1} | \theta_1\right), \log P\left({\mathbf{x} \choose 2} | \theta_2\right), \cdots, \log P\left({\mathbf{x} \choose d} | \theta_d\right)  \right] \in {R}^D
    \end{equation}
    $$
    where term $P\left({{x}} | \theta_d\right)$ in the above expansion denotes the probability density function of the multivariate normal distribution with hyper-parameter $\theta_d$,
    $$
        \begin{equation}
            P\left(x | \theta_d\right) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}),
        \end{equation}
    $$
    where the hyper-parameter $\theta_d = (\mathbf{\mu}, \mathbf{\Sigma})$ covers the mean vector $\mathbf{\mu}$ and variance matrix $\mathbf{\Sigma}$.

    For combinatorial normal probabilistic expansion, its output expansion dimensions will be $D = \sum_{i=1}^d {m \choose i}$,
    where $d$ denotes the combinatorial expansion order parameter.

    By default, the input and output can also be processed with the optional pre- or post-processing functions
    in the gaussian rbf expansion function.

    Attributes
    ----------
    name: str, default = 'combinatorial_normal_expansion'
        The name of the combinatorial normal expansion function.
    d: int, default = 2
        The combinatorial expansion order.
    with_replacement: bool, default = False
        The with_replacement tag for the random combination.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(self, name: str = 'combinatorial_normal_expansion', d: int = 1, with_replacement: bool = False, *args, **kwargs):
        r"""
        The initialization method of the combinatorial normal probabilistic expansion function.

        It initializes a combinatorial normal probabilistic expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'combinatorial_normal_expansion'
            The name of the combinatorial normal probabilistic expansion function.
        d: int, default = 2
            The order of random combinations.
        with_replacement: bool, default = False
            The replacement boolean tag.
        """
        print('combinatorial_normal_expansion initialization')
        super().__init__(name=name, d=d, with_replacement=with_replacement, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the combinatorial expansion function, the expansion space dimension will be
        $$ D = \sum_{i=1}^d {m \choose i}. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        assert type(self.d) is int and self.d >= 1
        return int(sum([comb(m, r) for r in range(1, self.d+1)]))

    def forward(self, x: torch.Tensor, device='cpu', with_replacement: bool = False, *args, **kwargs):
        r"""
        The forward method of the combinatorial normal probabilistic expansion function.

        It performs the combinatorial data expansion of the input data and returns the expansion result as
        $$
        \begin{equation}
            \kappa(\mathbf{x}) = \left[ {\mathbf{x} \choose 1}, {\mathbf{x} \choose 2}, \cdots, {\mathbf{x} \choose d} \right] \in {R}^D.
        \end{equation}
        $$


        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.
        with_replacement: bool, default = False
            The replacement boolean tag.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        x = self.pre_process(x=x, device=device)
        expansion_shape = list(x.size())
        expansion_shape[-1] = self.calculate_D(m=expansion_shape[-1])

        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        if len(x.shape) >= 3:
            raise ValueError("Input x can only be 2d or 1d, higher dimensional inputs are not supported yet...")

        x = x.to('cpu')
        with_replacement = with_replacement if with_replacement is not None else self.with_replacement
        combinations = self.combinatorial(x=x, d=self.d, device=device, with_replacement=with_replacement, *args, **kwargs)

        result = torch.zeros(x.size(0), self.calculate_D(m=x.size(1)))
        distribution_dict = {}
        current_index = 0
        for r in range(1, self.d+1):
            if r not in distribution_dict:
                # multivariate_normal_distributions
                distribution_dict[r] = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=torch.zeros(r), covariance_matrix=torch.eye(r)
                )
            degree_batch_expansion = torch.stack(combinations[r-1], dim=0)
            tuple_count = len(degree_batch_expansion[0])
            degree_batch_log_likelihood = distribution_dict[r].log_prob(value=degree_batch_expansion)
            result[:, current_index:current_index+tuple_count] = degree_batch_log_likelihood
            current_index += tuple_count
        return self.post_process(x=result.view(*expansion_shape), device=device).to(device)



if __name__ == "__main__":
    import time
    #start_time = time.time()
    #a = torch.Tensor([1, 2, 3, 4])
    #b = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    #rcf = combinatorial_normal_expansion(d=4)#, postprocess_functions='layer_norm', postprocess_functions_parameters={'dim': 1})
    #print(rcf(b, device='mps'), time.time()-start_time)

    #a = torch.Tensor(['SepalLengthCm_float', 'SepalWidthCm_float', 'PetalLengthCm_float', 'PetalWidthCm_float'])
    a = torch.Tensor([1, 2, 3, 4])
    exp_func = combinatorial_expansion(d=4)
    print(exp_func(x=a))