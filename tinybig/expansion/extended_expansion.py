# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
extended data expansion functions.

This module contains the extended data expansion function, which concatenates multiple expansion functions
for defining more complex data expansions.
"""

import torch.nn

from tinybig.expansion import transformation
from tinybig.util.util import get_obj_from_str

#####################
# Nested Expansions #
#####################


class extended_expansion(transformation):
    r"""
    The extended data expansion function.

    It performs the data expansion of multiple expansion functions, and conctatnates their expansions to define
    wider expansions of the input data vector.

    ...

    Notes
    ----------
    Formally, given the $n$ different expansion functions (1) $\kappa_1: {R}^{m} \to {R}^{d_1}$,
    (2) $\kappa_2: {R}^{m} \to {R}^{d_2}$, $\cdots$, (n) $\kappa_n: {R}^{m} \to {R}^{d_{n}}$,
    we can represent their extended data expansion function $\kappa: {R}^{m} \to {R}^D$ as follows:
    $$
        \begin{equation}
        \kappa(\mathbf{x}) = \left[ \kappa_1\left( \mathbf{x} \right), \kappa_2\left( \mathbf{x} \right), \cdots, \kappa_n\left( \mathbf{x} \right) \right] \in {R}^{D},
        \end{equation}
    $$
    where the expansion output dimension $D = \sum_{i=1}^n d_i$.

    Attributes
    ----------
    name: str, default = 'extended_expansion'
        The name of the extended expansion function.
    expansion_functions: list
        The list of expansion functions to be extended.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    forward
        It implements the abstract forward method declared in the base expansion class.

    """
    def __init__(
            self,
            name='extended_expansion',
            expansion_functions: list = None,
            expansion_function_configs: list | dict = None,
            *args, **kwargs
    ):
        r"""
        The initialization method of the extended expansion function.

        It initializes an extended expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'extended_expansion'
            The name of the extended expansion function.
        expansion_functions: list, default = None
            The list of data expansion functions to be extended.
        expansion_function_configs: list | dict, default = None
            The list or dictionary of the expansion function configurations.

        """
        super().__init__(name=name, *args, **kwargs)
        self.expansion_functions = []
        if expansion_functions is not None and expansion_functions != []:
            self.expansion_functions = expansion_functions
        elif expansion_function_configs is not None:
            for expansion_config in expansion_function_configs:
                expansion_class = expansion_config['expansion_class']
                expansion_parameters = expansion_config['expansion_parameters']
                self.expansion_functions.append(get_obj_from_str(expansion_class)(**expansion_parameters))

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate expansion space dimension based on the input dimension parameter m.
        For the extended expansion function defined based on $n$ functions (1) $\kappa_1: {R}^{m} \to {R}^{d_1}$,
        (2) $\kappa_2: {R}^{m} \to {R}^{d_2}$, $\cdots$, (n) $\kappa_n: {R}^{m} \to {R}^{d_{n}}$,
        the expansion space dimension will be
        $$ D = \sum_{i=1}^n d_i. $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        D = 0
        for expansion_func in self.expansion_functions:
            D += expansion_func.calculate_D(m=m)
        return D

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the extended expansion function.

        For the extended expansion function defined based on $n$ functions (1) $\kappa_1: {R}^{m} \to {R}^{d_1}$,
        (2) $\kappa_2: {R}^{m} \to {R}^{d_2}$, $\cdots$, (n) $\kappa_n: {R}^{m} \to {R}^{d_{n}}$,
        it performs the extended expansion of the input data and returns the expansion result as
        $$
            \begin{equation}
                \kappa(\mathbf{x}) = \left[ \kappa_1\left( \mathbf{x} \right), \kappa_2\left( \mathbf{x} \right), \cdots, \kappa_n\left( \mathbf{x} \right) \right] \in {R}^{D}.
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
        x = self.pre_process(x=x)
        expansion = []
        for expansion_func in self.expansion_functions:
            expansion.append(expansion_func(x=x, device=device))
        return torch.cat(expansion, dim=1)