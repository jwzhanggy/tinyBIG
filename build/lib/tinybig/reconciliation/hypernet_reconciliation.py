# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Hypernet based parameter reconciliation functions.

This module contains the hypernet based parameter reconciliation functions,
including hypernet_reconciliation.
"""

import warnings
import torch
import torch.nn as nn

from tinybig.reconciliation import fabrication


###########################
# Hypernet reconciliation #
###########################

class hypernet_reconciliation(fabrication):
    r"""
    The hypernet based parameter reconciliation function.

    It performs the hypernet based parameter reconciliation, and returns the reconciled parameter matrix of shape (n, D).
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    ...

    Notes
    ----------
    Formally, given the input parameter vector $\mathbf{w} \in {R}^l$ from length $l$, the hypernet based
    parameter reconciliation function projects it to a high-dimensional parameter matrix of shape (n, D) via a hypernet
    model, e.g., MLP, as follows
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \text{HyperNet}(\mathbf{w}) = \mathbf{W} \in {R}^{n \times D},
        \end{equation}
    $$
    where $\text{HyperNet}(\cdot)$ denotes a randomly initialized MLP model with frozen parameters.

    For the hybernet based parameter reconciliation function, **the parameter length $l$ should be assigned manually** in
    the initialization method, and it cannot be calculated based on the dimension parameters $n$ and $D$ anymore.

    Also in the current project, we use a frozen MLP with 1 hidden layer as the hypernet for parameter reconciliation.
    Meanwhile, the current implementation of this reconciliation function also allows the dynamic MLP with learnable
    parameters, which can be turned on or turned off by chanting the "static" parameter as True or False, respectively.

    Attributes
    ----------
    name: str, default = 'hypernet_reconciliation'
        Name of the hypernet parameter reconciliation function
    r: int, default = 2
        Submatrix rank parameter.

    Methods
    ----------
    __init__
        It initializes the hypernet parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters for the reconciliation function.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name='hypernet_reconciliation', l: int = 64, hidden_dim: int = 128, static: bool = True, net = None, *args, **kwargs):
        """
        The initialization method of the hypernet parameter reconciliation function.

        It initializes a hypernet parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'hypernet_reconciliation'
            Name of the hypernet based parameter reconciliation function.
        l: int, default = 64
            The learnable parameter length, which needs to be assigned manually.
        hidden_dim: int, default = 128
            The hidden layer dimension of the hypernet MLP.
        static: bool, default = True
            The static hypernet indicator. If state=True, the hypernet MLP is frozen; if state=False,
            the hypernet MLP is dynamic and contains learnable parameters as well.
        net: torch.nn.Sequential, default = None,
            The hypernet MLP model.

        Returns
        ----------
        object
            The hypernet parameter reconciliation function object.
        """
        super().__init__(name=name, *args, **kwargs)
        self.l = l
        self.hidden_dim = hidden_dim
        warnings.warn('In hypernet based reconciliation function, parameter l and hidden_dim cannot be None, '
                      'which will be set with the default values 64 and 128, respectively...')
        self.net = net
        self.static = static

    def calculate_l(self, n: int = None, D: int = None):
        r"""
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., $l$, of the parameter reconciliation function.

        Notes
        -------
        For the hybernet based parameter reconciliation function, **the parameter length $l$ should be assigned manually**
        in the initialization method, and it cannot be calculated based on the dimension parameters $n$ and $D$ anymore.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.

        Returns
        -------
        int
            The number of required learnable parameters.
        """
        return self.l

    def initialize_hypernet(self, l: int, n: int, D: int, hidden_dim: int, static: bool = True, device: str = 'cpu'):
        r"""
        The hypernet MLP initialization method.

        It initializes the hypernet MLP model based on the provided parameters, whose architecture dimensions
        can be denoted as follows:
        $$
            \begin{equation}
                [l] \to [hidden\\_dim] \to [n \times D],
            \end{equation}
        $$
        which can projects any inputs of length $l$ to the desired output of length $n \times D$.

        Parameters
        ----------
        l: int
            The input dimension of the hypernet MLP, which equals to the parameter length $l$.
        n: int
            The output space dimension, which together with the expansion dimension $D$ defines the output dimension of the bypernet MLP as $n \times D$.
        D: int
            The expansion space dimension, which together with the output space dimension $n$ defines the output dimension of the bypernet MLP as $n \times D$.
        hidden_dim: int
            The hidden layer dimension of the hypernet MLP.
        static: bool, default = True
            The static hypernet indicator. If state=True, the hypernet MLP is frozen; if state=False,
            the hypernet MLP is dynamic and contains learnable parameters as well.

        device: str, default = 'cpu'
            The device to host the hypernet and perform the parameter reconciliation.

        Returns
        -------
        None
            This function initialize the self.net parameter and doesn't have any return values.
        """
        self.net = nn.Sequential(
            nn.Linear(l, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, n*D)
        ).to(device)

        for param in self.net.parameters():
            torch.nn.init.normal_(param)

        if static:
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.net.parameters():
                param.detach()
        else:
            for param in self.net.parameters():
                param.requires_grad = True

    # def forwardtest(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
    #     print(w.shape, self.calculate_l(n, D))
    #     return F.linear(w, torch.ones(n*D, self.calculate_l(n, D)).to(device)).view(n, D).to(device)

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the hypernet based parameter reconciliation operation to the input parameter vector $\mathbf{w}$,
        and returns the reconciled parameter matrix of shape (n, D) subject to rank parameters $r$ as follows:
        $$
            \begin{equation}
                \psi(\mathbf{w}) = \text{HyperNet}(\mathbf{w}) = \mathbf{W} \in {R}^{n \times D},
            \end{equation}
        $$
        where $\text{HyperNet}(\cdot)$ denotes a randomly initialized MLP model with frozen parameters.

        Parameters
        ----------
        n: int
            The dimension of the output space.
        D: int
            The dimension of the intermediate expansion space.
        w: torch.nn.Parameter, default = None
            The learnable parameters of the model.
        device: str, default = 'cpu'
            Device to perform the parameter reconciliation.

        Returns
        ----------
        torch.Tensor
            The reconciled parameter matrix of shape (n, D).
        """
        assert w.dim() == 2 and w.size(1) == self.calculate_l(n=n, D=D)
        if self.net is None:
            self.initialize_hypernet(l=self.calculate_l(n, D), n=n, D=D, hidden_dim=self.hidden_dim, static=self.static, device=device)
        return self.net(w).view(n, D)
