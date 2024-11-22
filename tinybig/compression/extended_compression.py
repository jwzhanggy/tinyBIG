# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#########################
# Extended Compressions #
#########################

"""
Extended data compression functions.

This module contains the extended data compression functions
"""

from tinybig.expansion import extended_expansion


class extended_compression(extended_expansion):
    r"""
        The extended data compression function.

        It performs the data compression of multiple compression functions, and conctatnates their compressions to define
        wider compressions of the input data vector.
        This class inherits from the extended_expansion class.

        ...

        Notes
        ----------
        Formally, given the $n$ different compression functions (1) $\kappa_1: {R}^{m} \to {R}^{d_1}$,
        (2) $\kappa_2: {R}^{m} \to {R}^{d_2}$, $\cdots$, (n) $\kappa_n: {R}^{m} \to {R}^{d_{n}}$,
        we can represent their extended data compression function $\kappa: {R}^{m} \to {R}^D$ as follows:
        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \left[ \kappa_1\left( \mathbf{x} \right), \kappa_2\left( \mathbf{x} \right), \cdots, \kappa_n\left( \mathbf{x} \right) \right] \in {R}^{D},
            \end{equation}
        $$
        where the compression output dimension $D = \sum_{i=1}^n d_i$.

        Attributes
        ----------
        name : str, optional
            The name of the nested compression function. Defaults to 'extended_compression'.

        Methods
        ----------
        __init__
            It performs the initialization of the compression function.

    """
    def __init__(self, name='extended_compression', *args, **kwargs):
        r"""
            The initialization method of the extended compression function.

            It initializes an extended compression object based on the input function name.
            This method will also call the initialization method of the base class as well.

            Parameters
            ----------
            name : str, optional
                The name of the nested compression function. Defaults to 'extended_compression'.

            Returns
            ----------
            transformation
                The extended data compression function.
        """
        super().__init__(name=name, *args, **kwargs)
