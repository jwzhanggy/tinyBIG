# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# Nested Compressions #
#######################

from tinybig.expansion import nested_expansion


class nested_compression(nested_expansion):
    r"""
        The nested data compression function.

        It performs the data compression of multiple compression functions, and conctatnates their compressions to define
        wider compressions of the input data vector.
        This class inherits from the nested_expansion class.

        ...

        Notes
        ----------
        Formally, given the $n$ different compression functions (1) $\kappa_1: {R}^{m} \to {R}^{d_1}$,
            (2) $\kappa_2: {R}^{d_1} \to {R}^{d_2}$, $\cdots$, (n) $\kappa_n: {R}^{d_{n-1}} \to {R}^{D}$,
        we can represent their nested data compression function $\kappa: {R}^{m} \to {R}^D$ as follows:
        $$
            \begin{equation}
                \kappa(\mathbf{x}) = \kappa_{n} \left( \kappa_{n-1} \left( \cdots \kappa_2 \left( \kappa_{1} \left( \mathbf{x} \right) \right) \right) \right) \in {R}^{D}.
            \end{equation}
        $$
        where the compression output dimension $D$ is determined by the last compression function $\kappa_n$.

        Attributes
        ----------
        name : str, optional
            The name of the nested compression function. Defaults to 'nested_compression'.

        Methods
        ----------
        __init__
            It performs the initialization of the compression function.

    """

    def __init__(self, name='nested_compression', *args, **kwargs):
        r"""
            The initialization method of the nested compression function.

            It initializes an nested compression object based on the input function name.
            This method will also call the initialization method of the base class as well.

            Parameters
            ----------
            name : str, optional
                The name of the nested compression function. Defaults to 'nested_compression'.

            Returns
            ----------
            transformation
                The nested data compression function.
        """
        super().__init__(name=name, *args, **kwargs)
