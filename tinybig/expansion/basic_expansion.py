# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Data expansion basic functions.

This module contains the basic data expansion functions, including identity_expansion, reciprocal_expansion and linear_expansion.
"""

import torch.nn

from tinybig.expansion import expansion

####################
# Basic Expansions #
####################


class identity_expansion(expansion):
    """
    class identity_expansion

    The identity data expansion function.

    The class inherits from the base expansion class.

    ...

    Attributes
    ----------
    name: str, default = 'identity_expansion'
        Name of the expansion function.

    Methods
    ----------
    __init__
        It performs the initialization of the expansion function.

    calculate_D
        It calculates the expansion space dimension D based on the input dimension parameter m.

    __call__
        It reimplements the abstract callable method declared in the base expansion class.

    """
    def __init__(self, name='identity_expansion', *args, **kwargs):
        """
        The initialization method of the identity expansion function.

        It initializes an identity expansion object based on the input metric name.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'identity_expansion'
            The name of the identity expansion function.
        """
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        r"""
        The expansion dimension calculation method.

        It calculates the intermediate transformation space dimension based on the input dimension parameter m.

        Notes
        -----
        For the identity expansion function, the expansion space dimension equals to the input space dimension, i.e.,

        $$ D = m $$

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the transformation space.
        """
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The callable method of the data transformation function.

        It performs the identity data expansion of the input data and returns the expansion result.

        Notes
        ----------
        For the identity expansion function, the expansion space dimension equals to the input space dimension.

        In other words, for any input $\mathbf{x} \in R^m$, its identity expansion will be
        $$
            \begin{equation}
                \kappa(\mathbf{x}) = \sigma(\mathbf{x}) \in R^D
            \end{equation}
        $$
        where $D = m$ and $\sigma(\cdot)$ denotes the optional pre-processing functions.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data transformation.
        args: list, default = ()
            The other parameters.
        kwargs: dict, default = {}
            The other parameters.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        x = self.pre_process(x=x, device=device)
        expansion = x
        return self.post_process(x=expansion, device=device)


class reciprocal_expansion(expansion):
    def __init__(self, name='reciprocal_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x[torch.logical_and(x>=0, x<=1e-6)] = 1e-6
        x[torch.logical_and(x<0, x>=-1e-6)] = -1e-6
        expansion = torch.reciprocal(x)
        return self.post_process(x=expansion, device=device)


class linear_expansion(expansion):
    def __init__(self, name='linear_expansion', c=None, pre_C=None, post_C=None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.c = c
        self.pre_C = pre_C
        self.post_C = post_C

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', c=None, pre_C=None, post_C=None, *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        c = c if c is not None else self.c
        pre_C = pre_C if pre_C is not None else self.pre_C
        post_C = post_C if post_C is not None else self.post_C

        if c is not None:
            expansion = c * x
        elif pre_C is not None:
            assert pre_C.size(-1) == x.size(0)
            expansion = torch.matmul(pre_C, x)
        elif post_C is not None:
            assert x.size(-1) == post_C.size(0)
            expansion = torch.matmul(x, post_C)
        else:
            expansion = x

        return self.post_process(x=expansion, device=device)


if __name__== '__main__':
    import torch.nn.functional as F
    exp = reciprocal_expansion(postprocess_functions=F.sigmoid)
    x = torch.Tensor([[0.5, 0.5]])
    print(x, exp(x))