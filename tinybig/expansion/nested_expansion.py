# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
nested data expansion functions.

This module contains the nested data expansion function, which concatenates multiple expansion functions
for defining more complex data expansions.
"""

import torch.nn

from tinybig.expansion import transformation
from tinybig.util.util import get_obj_from_str

#####################
# Nested Expansions #
#####################


class nested_expansion(transformation):
    r"""
    The nested data expansion function.

    It performs the data expansion of multiple expansion functions, and conctatnates their expansions to define
    wider expansions of the input data vector.

    ...

    Notes
    ----------
    Formally, given the $n$ different expansion functions (1) $\kappa_1: {R}^{m} \to {R}^{d_1}$,
        (2) $\kappa_2: {R}^{d_1} \to {R}^{d_2}$, $\cdots$, (n) $\kappa_n: {R}^{d_{n-1}} \to {R}^{D}$,
    we can represent their nested data expansion function $\kappa: {R}^{m} \to {R}^D$ as follows:
    $$
        \begin{equation}
            \kappa(\mathbf{x}) = \kappa_{n} \left( \kappa_{n-1} \left( \cdots \kappa_2 \left( \kappa_{1} \left( \mathbf{x} \right) \right) \right) \right) \in {R}^{D}.
        \end{equation}
    $$
    where the expansion output dimension $D$ is determined by the last expansion function $\kappa_n$.

    Attributes
    ----------
    name: str, default = 'nested_expansion'
        The name of the nested expansion function.
    expansion_functions: list
        The list of expansion functions to be nested.

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
            name='nested_expansion',
            expansion_functions: list = None,
            expansion_function_configs: list | dict = None,
            *args, **kwargs
    ):
        r"""
        The initialization method of the nested expansion function.

        It initializes an nested expansion object based on the input function name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'nested_expansion'
            The name of the nested expansion function.
        expansion_functions: list, default = None
            The list of data expansion functions to be nested.
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
        For the nested expansion function defined based on $n$ functions (1) $\kappa_1: {R}^{m} \to {R}^{d_1}$,
        (2) $\kappa_2: {R}^{d_1} \to {R}^{d_2}$, $\cdots$, (n) $\kappa_n: {R}^{d_{n-1}} \to {R}^{D}$,
        the expansion space dimension will be determined by the last expansion function $\kappa_n$.

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the expansion space.
        """
        D = m
        for expansion_func in self.expansion_functions:
            D = expansion_func.calculate_D(m=D)
        return D

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the nested expansion function.

        For the nested expansion function defined based on $n$ functions (1) $\kappa_1: {R}^{m} \to {R}^{d_1}$,
        (2) $\kappa_2: {R}^{d_1} \to {R}^{d_2}$, $\cdots$, (n) $\kappa_n: {R}^{d_{n-1}} \to {R}^{D}$,
        it performs the nested expansion of the input data and returns the expansion result as
        $$
            \begin{equation}
                \kappa(\mathbf{x}) = \kappa_{n} \left( \kappa_{n-1} \left( \cdots \kappa_2 \left( \kappa_{1} \left( \mathbf{x} \right) \right) \right) \right) \in {R}^{D}.
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
        expansion = x
        for expansion_func in self.expansion_functions:
            expansion = expansion_func(x=expansion, device=device)
        return expansion