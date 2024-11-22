# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################################
# Random Matrix based reconciliations #
#######################################

"""
Low-rank parameter reconciliation functions.

This module contains the low-rank parameter reconciliation functions,
including lorr_reconciliation, hm_reconciliation, lphm_reconciliation, and dual_lphm_reconciliation.
"""

import torch
import torch.nn.functional as F

from tinybig.reconciliation import fabrication


class random_matrix_adaption_reconciliation(fabrication):
    r"""
        A reconciliation mechanism using random matrices for parameter adaptation.

        This class generates a reconciliation matrix `W` based on random matrices and diagonal parameter matrices.

        Notes
        ----------

        Formally, given a parameter vector $\mathbf{w} \in R^l$ of length $l$, we can partition it into two vectors $\lambda_1 \in R^{n}$ and $\lambda_2 \in R^r$.
        These two vectors will define two diagonal matrices $\Lambda_1 = diag( \lambda_1) \in R^{n \times n}$ and $\Lambda_2 = diag(\lambda_2) \in R^{r \times r}$.

        These two sub-matrices will fabricate a parameter matrix of shape $n \times D$ as follows:

        $$
            \begin{equation}
            \xi(\mathbf{w}) =  \Lambda_1 \mathbf{A} \Lambda_1 \mathbf{B}^\top \in R^{n \times D},
            \end{equation}
        $$

        where matrices $\mathbf{A} \in R^{n \times r}$ and $\mathbf{B} \in R^{D \times r}$ are randomly sampled from the Gaussian distribution $\mathcal{N}(\mathbf{0}, \mathbf{I})$.
        The required length of vector $\mathbf{w}$ is $l = n + r$.

        Attributes
        ----------
        r : int
            Rank of the random matrices used in the adaptation.
        A : torch.Tensor
            Random matrix of shape `(n, r)`, initialized and reused during computation.
        B : torch.Tensor
            Random matrix of shape `(D, r)`, initialized and reused during computation.

        Methods
        -------
        calculate_l(n, D)
            Computes the number of parameters required for the reconciliation.
        forward(n, D, w, device='cpu', *args, **kwargs)
            Computes the reconciliation matrix using the provided parameters and random matrices.
    """

    def __init__(self, name: str = 'random_matrix_adaption_reconciliation', r: int = 2, *args, **kwargs):
        """
            Initializes the random matrix adaption reconciliation mechanism.

            Parameters
            ----------
            name : str, optional
                Name of the reconciliation instance. Defaults to 'random_matrix_adaption_reconciliation'.
            r : int, optional
                Rank of the random matrices used in the adaptation. Defaults to 2.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.A = None
        self.B = None

    def calculate_l(self, n: int, D: int):
        """
            Computes the number of parameters required for the reconciliation.

            Parameters
            ----------
            n : int
                Number of rows in the reconciliation matrix.
            D : int
                Number of columns in the reconciliation matrix.

            Returns
            -------
            int
                Total number of parameters required, which is `n + r`.
        """
        return n + self.r

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        """
            Computes the reconciliation matrix using the provided parameters and random matrices.

            Parameters
            ----------
            n : int
                Number of rows in the reconciliation matrix.
            D : int
                Number of columns in the reconciliation matrix.
            w : torch.nn.Parameter
                Parameter tensor of shape `(n, n + r)`, where `r` is the rank of the random matrices.
            device : str, optional
                Device for computation ('cpu', 'cuda', etc.). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Reconciliation matrix of shape `(n, D)`.

            Raises
            ------
            AssertionError
                If the dimensions of `w`, `A`, or `B` are inconsistent with the expected shapes.
        """
        assert w.ndim == 2 and w.numel() == self.calculate_l(n=n, D=D)
        lambda_1, lambda_2 = torch.split(w, [n, self.r], dim=1)

        Lambda_1 = torch.diag(lambda_1.view(-1)).to(device)
        Lambda_2 = torch.diag(lambda_2.view(-1)).to(device)

        if self.A is None or (self.A is not None and self.A.shape != (n, self.r)):
            self.A = torch.randn(n, self.r, device=device)
        if self.B is None or (self.B is not None and self.B.shape != (D, self.r)):
            self.B = torch.randn(D, self.r, device=device)
        assert self.A.shape == (n, self.r) and self.B.shape == (D, self.r)

        W = torch.matmul(torch.matmul(torch.matmul(Lambda_1, self.A), Lambda_2), self.B.t())
        assert W.shape == (n, D)
        return W


class random_matrix_hypernet_reconciliation(fabrication):
    r"""
        A reconciliation mechanism using a hypernetwork approach with random matrices.

        This class computes a reconciliation matrix `W` using a series of random matrices and a hypernetwork-like architecture.

        Notes
        ----------

        Formally, based on the parameter vector $\mathbf{w} \in R^l$, the random_matrix_hypernet_reconciliation function will fabricate it into a parameter matrix $\mathbf{W} \in R^{n \times D}$ as follow:

        $$
            \begin{equation}
            \begin{aligned}
            \text{Hypernet}(\mathbf{w}) &= \sigma(\mathbf{w} \mathbf{H}_1) \mathbf{H}_2 \\
            &= \sigma \left( \mathbf{w} (\mathbf{P} \mathbf{Q}^\top) \right) \left( \mathbf{S} \mathbf{T}^\top \right)\\
            &= \left( \sigma \left( (\mathbf{w} \mathbf{P}) \mathbf{Q}^\top \right) \mathbf{S} \right) \mathbf{T}^\top \in R^{n \times D},
            \end{aligned}
            \end{equation}
        $$

        where $\mathbf{P} \in R^{l \times r}$, $\mathbf{Q} \in R^{d \times r}$, $\mathbf{S} \in R^{d \times r}$ and $\mathbf{T} \in R^{(n \times D) \times r}$
        are the low-rank random and frozen sub-matrices that can compose the matrices $\mathbf{H}_1 \in R^{l \times d}$ and $\mathbf{H}_2 \in R^{d \times (n \times D)}$ of the hypernet.
        Moreover, by leveraging the associative law of matrix multiplication, we can avoid explicitly calculating and storing $\mathbf{H}_1$ and $\mathbf{H}_2$ as indicated by the above equation.
        These low-rank random matrix representations reduce the space consumption of this function to $\mathcal{O}\left(r \cdot (l + 2d + n \cdot D)\right)$.

        Attributes
        ----------
        r : int
            Rank of the random matrices.
        l : int
            Dimension of the hypernetwork input.
        hidden_dim : int
            Hidden dimension of the hypernetwork.
        P : torch.Tensor
            Random matrix of shape `(l, r)`, initialized and reused during computation.
        Q : torch.Tensor
            Random matrix of shape `(hidden_dim, r)`, initialized and reused during computation.
        S : torch.Tensor
            Random matrix of shape `(hidden_dim, r)`, initialized and reused during computation.
        T : torch.Tensor
            Random matrix of shape `(n * D, r)`, initialized and reused during computation.

        Methods
        -------
        calculate_l(n=None, D=None)
            Computes the number of parameters required for the hypernetwork.
        forward(n, D, w, device='cpu', *args, **kwargs)
            Computes the reconciliation matrix using the hypernetwork approach.
    """
    def __init__(self, name='random_matrix_hypernet_reconciliation', r: int = 2, l: int = 64, hidden_dim: int = 128, *args, **kwargs):
        """
            Initializes the random matrix hypernetwork reconciliation mechanism.

            Parameters
            ----------
            name : str, optional
                Name of the reconciliation instance. Defaults to 'random_matrix_hypernet_reconciliation'.
            r : int, optional
                Rank of the random matrices. Defaults to 2.
            l : int, optional
                Dimension of the hypernetwork input. Defaults to 64.
            hidden_dim : int, optional
                Hidden dimension of the hypernetwork. Defaults to 128.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.l = l
        self.hidden_dim = hidden_dim

        self.P = None
        self.Q = None
        self.S = None
        self.T = None

    def calculate_l(self, n: int = None, D: int = None):
        """
            Computes the number of parameters required for the hypernetwork.

            Parameters
            ----------
            n : int, optional
                Number of rows in the reconciliation matrix (unused here). Defaults to None.
            D : int, optional
                Number of columns in the reconciliation matrix (unused here). Defaults to None.

            Returns
            -------
            int
                Total number of parameters required, equal to `l`.
        """
        assert self.l is not None
        return self.l

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        """
            Computes the reconciliation matrix using the hypernetwork approach.

            Parameters
            ----------
            n : int
                Number of rows in the reconciliation matrix.
            D : int
                Number of columns in the reconciliation matrix.
            w : torch.nn.Parameter
                Parameter tensor of shape `(1, l)` where `l` is the hypernetwork input dimension.
            device : str, optional
                Device for computation ('cpu', 'cuda', etc.). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Reconciliation matrix of shape `(n, D)`.

            Raises
            ------
            AssertionError
                If the dimensions of `w`, `P`, `Q`, `S`, or `T` are inconsistent with the expected shapes.
        """

        assert w.ndim == 2 and w.numel() == self.calculate_l(n=n, D=D)

        if self.P is None or (self.P is not None and self.P.shape != (self.l, self.r)):
            self.P = torch.randn(self.l, self.r, device=device)
        if self.Q is None or (self.Q is not None and self.Q.shape != (self.hidden_dim, self.r)):
            self.Q = torch.randn(self.hidden_dim, self.r, device=device)
        assert self.P.shape == (self.l, self.r) and self.Q.shape == (self.hidden_dim, self.r)

        if self.S is None or (self.S is not None and self.S.shape != (self.hidden_dim, self.r)):
            self.S = torch.randn(self.hidden_dim, self.r, device=device)
        if self.T is None or (self.T is not None and self.T.shape != (n*D, self.r)):
            self.T = torch.randn(n*D, self.r, device=device)
        assert self.S.shape == (self.hidden_dim, self.r) and self.T.shape == (n*D, self.r)

        W = torch.matmul(
            torch.matmul(
                F.sigmoid(torch.matmul(torch.matmul(w, self.P), self.Q.t())),
                self.S),
            self.T.t()
        ).view(n, D)

        assert W.shape == (n, D)
        return W

