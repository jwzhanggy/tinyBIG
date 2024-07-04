# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Low-rank parameter reconciliation functions.

This module contains the low-rank parameter reconciliation functions,
including lorr_reconciliation, hm_reconciliation, lphm_reconciliation, and dual_lphm_reconciliation.
"""

import torch
import torch.nn.functional as F

from tinybig.reconciliation import fabrication


############################
# Low-rank reconciliations #
############################


class lorr_reconciliation(fabrication):
    r"""
    The low-rank parameter reconciliation function.

    It performs the low-rank parameter reconciliation, and returns the low-rank reconciled parameter matrix of shape (n, D).
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    ...

    Notes
    ----------
    Formally, given the parameter vector $\mathbf{w} \in {R}^{l}$ and a rank hyper-parameter $r$,
    low-rank parameter reconciliation function partitions $\mathbf{w}$ into two sub-vectors and subsequently
    reshapes them into two matrices $\mathbf{A} \in {R}^{n \times r}$ and $\mathbf{B} \in {R}^{D \times r}$,
    each possessing a rank of $r$.
    These two sub-matrices $\mathbf{A}$ and $\mathbf{B}$ help define the low-rank reconciliation function as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \mathbf{B}^\top \in {R}^{n \times D}.
        \end{equation}
    $$
    In implementation, the matrix rank $r$ is defined as the hyper-parameter, which in turn determines the
    desired parameter length $l$ in accordance with the stated constraints.
    This necessitates imposing certain limitations on these dimension and rank parameters represented as follows:
    $$
        \begin{equation}
            l = (n + D) \times r.
        \end{equation}
    $$

    Attributes
    ----------
    name: str, default = 'low_rank_reconciliation'
        Name of the low-rank parameter reconciliation function
    r: int, default = 2
        Submatrix rank parameter.

    Methods
    ----------
    __init__
        It initializes the low-rank parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters for the reconciliation function.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name: str = 'low_rank_reconciliation', r: int = 2, *args, **kwargs):
        """
        The initialization method of the low-rank parameter reconciliation function.

        It initializes a low-rank parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'low_rank_reconciliation'
            Name of the low-rank parameter reconciliation function.
        r: int, default = 2
            Matrix rank parameter of the low-rank parameter reconciliation.

        Returns
        ----------
        object
            The low-rank parameter reconciliation function object.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r

    def calculate_l(self, n: int, D: int):
        r"""
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., $l$, of the parameter reconciliation function
        based on the intermediate and output space dimensions, $n$ and $D$, and the rank parameters $r$,
        which can be represented as follows:
        $$
            \begin{equation}
                l = (n + D) \times r.
            \end{equation}
        $$

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
        return self.r * (n + D)

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the low-rank parameter reconciliation operation to the input parameter vector $\mathbf{w}$,
        and returns the reconciled parameter matrix of shape (n, D) subject to rank parameters $r$ as follows:
        $$
            \begin{equation}
                \psi(\mathbf{w}) = \mathbf{A} \mathbf{B}^\top \in {R}^{n \times D}.
            \end{equation}
        $$
        where $\mathbf{A} \in {R}^{n \times r}$ and $\mathbf{B} \in {R}^{D \times r}$ are two low-rank matrices of rank
        $r$ obtained by partitioning $\mathbf{w}$ into two sub-vectors and subsequently reshaping them into matrices.

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
        A, B = torch.split(w, [self.r*n, self.r*D], dim=1)
        return F.linear(A.view(n, self.r), B.view(D, self.r))


class hm_reconciliation(fabrication):
    r"""
    The hypercomplex multiplication based parameter reconciliation function.

    It performs the hypercomplex multiplication based parameter reconciliation,
    and returns the reconciled parameter matrix of shape (n, D).
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    ...

    Notes
    ----------
    Formally, given the parameter vector $\mathbf{w}$ of length $l$ through partitioning and subsequent reshaping,
    we can create two parameter sub-matrices $\mathbf{A} \in R^{p \times q}$ and $\mathbf{B} \in R^{s \times t}$.
    The hypercomplex multiplication-based reconciliation computes the Kronecker product of these two parameter matrices
    to define the reconcilied parameter matrix of shape (n, D) as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} \in {R}^{n \times D},
        \end{equation}
    $$
    where the parameter dimension parameters should meeting the constraints that $l = pq + st$ and $n = ps$ and $D = qt$.

    In implementation, to reduce the number of hyper-parameters and accommodate the parameter dimensions,
    we can maintain the size of matrix $\mathbf{A}$ as fixed by two hyper-parameters $p$ and $q$, i.e.,
    $\mathbf{A} \in {R}^{p \times q}$.
    Subsequently, the desired size of matrix $\mathbf{B}$ can be directly calculated as $s \times t$,
    where $s =\frac{n}{p}$ and $t = \frac{D}{q}$.
    The hyper-parameters $p$ and $q$ need to be divisors of $n$ and $D$, respectively.
    Since both $\mathbf{A}$ and $\mathbf{B}$ originate from $\mathbf{w}$, the desired parameter length defining
    $\mathbf{w}$ can be obtained as
    $$
        \begin{equation}
            l = p \times q + \frac{n}{p} \times \frac{D}{q}.
        \end{equation}
    $$

    Attributes
    ----------
    name: str, default = 'hypercomplex_multiplication_reconciliation'
        Name of the hypercomplex multiplication based parameter reconciliation function
    p: int, default = 2
        Parameter sub-matrix row dimension.
    q: int, default = None
        Parameter sub-matrix column dimension.
        If q is not provided with initial values, it will be assigned with value p by default.

    Methods
    ----------
    __init__
        It initializes the hypercomplex multiplication based parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters for the reconciliation function.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name='hypercomplex_multiplication_reconciliation', p=2, q=None, *args, **kwargs):
        """
        The initialization method of the hypercomplex multiplication based parameter reconciliation function.

        It initializes a hypercomplex multiplication based parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'hypercomplex_multiplication_reconciliation'
            Name of the hypercomplex multiplication based parameter reconciliation function
        p: int, default = 2
            Parameter sub-matrix row dimension.
        q: int, default = None
            Parameter sub-matrix column dimension.
            If q is not provided with initial values, it will be assigned with value p by default.

        Returns
        ----------
        object
            The hypercomplex multiplication based parameter reconciliation function object.
        """
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p

    def calculate_l(self, n: int, D: int):
        r"""
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., $l$, of the parameter reconciliation function
        based on the intermediate and output space dimensions, $n$ and $D$, and the parameters $p$ and $q$,
        which can be represented as follows:
        $$
            \begin{equation}
                l = p \times q + \frac{n}{p} \times \frac{D}{q}.
            \end{equation}
        $$

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
        if n % self.p != 0 or D % self.q != 0:
            raise ValueError('The input dimensions {} and {} cannot be divided by parameter p {} and q {}'.format(n, D, self.p, self.q))
        s, t = int(n / self.p), int(D / self.q)
        assert (self.p * self.q * s * t == n * D)
        return self.p * self.q + s * t

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the hypercomplex multiplication based parameter reconciliation operation to the input parameter vector $\mathbf{w}$,
        and returns the reconciled parameter matrix of shape (n, D) subject to the parameters $p$ and $q$ as follows:
        $$
            \begin{equation}
                \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} \in {R}^{n \times D},
            \end{equation}
        $$
        where $\mathbf{A} \in {R}^{p \times q}$ and $\mathbf{B} \in {R}^{s \times t}$ are two sub-matrices of obtained
        by partitioning $\mathbf{w}$ into two sub-vectors and subsequently reshaping them into matrices.

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
        s, t = int(n/self.p), int(D/self.q)
        A, B = torch.split(w, [self.p*self.q, s*t], dim=1)
        return torch.einsum('pq,st->psqt', A, B).view(self.p*s, self.q*t)


class lphm_reconciliation(fabrication):
    r"""
    The low-rank parameterized hypercomplex multiplication (LPHM) based parameter reconciliation function.

    It performs the LPHM parameter reconciliation, and returns the LPHM reconciled parameter matrix of shape (n, D).
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    The low-rank parameterized hypercomplex multiplication based parameter reconciliation can be viewed as a combination
    of the low-rank parameter reconciliation with the hypercomplex multiplication based parameter reconciliation, where
    the matrix $\mathbf{B}$ in the hypercomplex multiplication based parameter reconciliation is replaced with the
    product of two low-rank sub-matrices instead.

    ...

    Notes
    ----------
    Formally, given the parameter vector $\mathbf{w} \in {R}^{l}$ and a rank hyper-parameter $r$, together with the
    parameter sub-matrix dimension parameters $p$ and $q$, the LPHM reconciliation function partitions $\mathbf{w}$
    into three sub-vectors and subsequently reshapes them into three matrices $\mathbf{A} \in {R}^{p \times q}$,
    $\mathbf{S} \in {R}^{\frac{n}{p} \times r}$ and $\mathbf{T} \in {R}^{\frac{D}{q} \times r}$.
    These sub-matrices $\mathbf{A}$, $\mathbf{S}$ and $\mathbf{T}$ help define the LPHM reconciliation function as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = \mathbf{A} \otimes ( \mathbf{S} \mathbf{T}^\top) \in {R}^{n \times D}.
        \end{equation}
    $$
    This necessitates imposing certain limitations on these dimension and rank parameters, and the parameter vector
    length $l$ can be calculated as follows:
    $$
        \begin{equation}
            l = p \times q + r( \frac{n}{p} + \frac{D}{p} ).
        \end{equation}
    $$

    For the LPHM parameter reconciliation function, it adds strict constraints on the parameters $p$ and $q$, which
    should be the divisors of the target dimensions $n$ and $D$, respectively, i.e.,
    $$
        \begin{equation}
            n \\% p = 0 \text{, and } D \\% q = 0.
        \end{equation}
    $$

    Attributes
    ----------
    name: str, default = 'lphm_reconciliation'
        Name of the LPHM parameter reconciliation function
    p: int, default = 2
        Parameter sub-matrix row dimension.
    q: int, default = None
        Parameter sub-matrix column dimension.
        If q is not provided with initial values, it will be assigned with value p by default.
    r: int, default = 2
        Submatrix rank parameter.

    Methods
    ----------
    __init__
        It initializes the LPHM parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters for the reconciliation function.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name='lphm_reconciliation', p=2, q=None, r=2, *args, **kwargs):
        """
        The initialization method of the LPHM parameter reconciliation function.

        It initializes a LPHM parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'lphm_reconciliation'
            Name of the LPHM parameter reconciliation function.
        p: int, default = 2
            Parameter sub-matrix row dimension.
        q: int, default = None
            Parameter sub-matrix column dimension.
            If q is not provided with initial values, it will be assigned with value p by default.
        r: int, default = 2
            Submatrix rank parameter.

        Returns
        ----------
        object
            The LPHM parameter reconciliation function object.
        """
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p
        self.r = r

    def calculate_l(self, n: int, D: int):
        r"""
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., $l$, of the parameter reconciliation function
        based on the intermediate and output space dimensions, $n$ and $D$, and the dimension and rank parameters
        $p$, $q$ and $r$, which can be represented as follows:
        $$
            \begin{equation}
                l = p \times q + r( \frac{n}{p} + \frac{D}{p} ).
            \end{equation}
        $$

        Notes
        ----------
        For the LPHM parameter reconciliation function, it adds strict constraints on the parameters $p$ and $q$, which
        should be the divisors of the target dimensions $n$ and $D$, respectively, i.e.,
        $$
            \begin{equation}
                n \\% p = 0 \text{, and } D \\% q = 0.
            \end{equation}
        $$

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
        if n % self.p != 0 or D % self.q != 0:
            raise ValueError('The input dimensions {} and {} cannot be divided by parameter p {} and q {}'.format(n, D, self.p, self.q))
        s, t = int(n / self.p), int(D / self.q)
        assert (self.p * self.q * s * t == n * D)
        return self.p * self.q + s * self.r + t * self.r

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the LPHM parameter reconciliation operation to the input parameter vector $\mathbf{w}$,
        and returns the reconciled parameter matrix of shape (n, D) subject to the dimension and rank parameters
        $p$, $q$ and $r$ as follows:
        $$
            \begin{equation}
                \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = \mathbf{A} \otimes ( \mathbf{S} \mathbf{T}^\top) \in {R}^{n \times D}.
            \end{equation}
        $$
        where $\mathbf{A} \in {R}^{p \times q}$, $\mathbf{S} \in {R}^{\frac{n}{p} \times r}$ and
        $\mathbf{T} \in {R}^{\frac{D}{q} \times r}$ are all obtained by partitioning $\mathbf{w}$ into sub-vectors
        and subsequently reshaping them into matrices.

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
        s, t = int(n/self.p), int(D/self.q)
        A, S, T = torch.split(w, [self.p*self.q, s*self.r, t*self.r], dim=1)
        B = F.linear(S.view(s, -1), T.view(t, -1)).view(1, -1)
        return torch.einsum('pq,st->psqt', A, B).view(self.p*s, self.q*t)


class dual_lphm_reconciliation(fabrication):
    r"""
    The dual low-rank parameterized hypercomplex multiplication (Dual-LPHM) based parameter reconciliation function.

    It performs the Dual-LPHM parameter reconciliation, and returns the Dual-LPHM reconciled parameter matrix of shape (n, D).
    This class inherits from the reconciliation class (i.e., the fabrication class in the module directory).

    The dual low-rank parameterized hypercomplex multiplication based parameter reconciliation can be viewed as a more
    agreesive version of the LPHM based parameter reconciliation function.
    It replaces both $\mathbf{A}$ and $\mathbf{B}$ in the hypercomplex multiplication based parameter reconciliation
    with the products of two low-rank sub-matrices, respectively.

    ...

    Notes
    ----------
    Formally, given the parameter vector $\mathbf{w} \in {R}^{l}$ and a rank hyper-parameter $r$, together with the
    parameter sub-matrix dimension parameters $p$ and $q$, the Dual-LPHM reconciliation function partitions $\mathbf{w}$
    into four sub-vectors and subsequently reshapes them into three matrices $\mathbf{P} \in {R}^{p \times r}$,
    $\mathbf{Q} \in {R}^{q \times r}$, $\mathbf{S} \in {R}^{\frac{n}{p} \times r}$ and $\mathbf{T} \in {R}^{\frac{D}{q} \times r}$.
    These sub-matrices $\mathbf{P}$, $\mathbf{Q}$, $\mathbf{S}$ and $\mathbf{T}$ help define the Dual-LPHM reconciliation function as follows:
    $$
        \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = ( \mathbf{P} \mathbf{Q}^\top) \otimes ( \mathbf{S} \mathbf{T}^\top) \in {R}^{n \times D}.
        \end{equation}
    $$
    This necessitates imposing certain limitations on these dimension and rank parameters, and the parameter vector
    length $l$ can be calculated as follows:
    $$
        \begin{equation}
            l = r( p + q + \frac{n}{p} + \frac{D}{p} ).
        \end{equation}
    $$

    For the Dual-LPHM parameter reconciliation function, it adds strict constraints on the parameters $p$ and $q$, which
    should be the divisors of the target dimensions $n$ and $D$, respectively, i.e.,
    $$
        \begin{equation}
            n \\% p = 0 \text{, and } D \\% q = 0.
        \end{equation}
    $$

    Attributes
    ----------
    name: str, default = 'dual_lphm_reconciliation'
        Name of the Dual-LPHM parameter reconciliation function
    p: int, default = 2
        Parameter sub-matrix row dimension.
    q: int, default = None
        Parameter sub-matrix column dimension.
        If q is not provided with initial values, it will be assigned with value p by default.
    r: int, default = 2
        Submatrix rank parameter.

    Methods
    ----------
    __init__
        It initializes the Dual-LPHM parameter reconciliation function.

    calculate_l
        It calculates the length of required parameters for the reconciliation function.

    forward
        It implements the abstract forward method declared in the base reconciliation class.
    """
    def __init__(self, name='dual_lphm_reconciliation', p=2, q=None, r=2, *args, **kwargs):
        """
        The initialization method of the Dual-LPHM parameter reconciliation function.

        It initializes a Dual-LPHM parameter reconciliation function object.
        This method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'dual_lphm_reconciliation'
            Name of the Dual-LPHM parameter reconciliation function.
        p: int, default = 2
            Parameter sub-matrix row dimension.
        q: int, default = None
            Parameter sub-matrix column dimension.
            If q is not provided with initial values, it will be assigned with value p by default.
        r: int, default = 2
            Submatrix rank parameter.

        Returns
        ----------
        object
            The Dual-LPHM parameter reconciliation function object.
        """
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p
        self.r = r

    def calculate_l(self, n: int, D: int):
        r"""
        The required parameter number calculation method.

        It calculates the number of required learnable parameters, i.e., $l$, of the parameter reconciliation function
        based on the intermediate and output space dimensions, $n$ and $D$, and the dimension and rank parameters
        $p$, $q$ and $r$, which can be represented as follows:
        $$
            \begin{equation}
                l = r( p + q + \frac{n}{p} + \frac{D}{p} ).
            \end{equation}
        $$

        Notes
        ----------
        For the Dual-LPHM parameter reconciliation function, it adds strict constraints on the parameters $p$ and $q$, which
        should be the divisors of the target dimensions $n$ and $D$, respectively, i.e.,
        $$
            \begin{equation}
                n \\% p = 0 \text{, and } D \\% q = 0.
            \end{equation}
        $$

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
        if n % self.p != 0 or D % self.q != 0:
            raise ValueError('The input dimensions {} and {} cannot be divided by parameter p {} and q {}'.format(n, D, self.p, self.q))
        s, t = int(n / self.p), int(D / self.q)
        assert (self.p * self.q * s * t == n * D)
        return self.p * self.r + self.q * self.r + s * self.r + t * self.r

    def forward(self, n: int, D: int, w: torch.nn.Parameter, device='cpu', *args, **kwargs):
        r"""
        The forward method of the parameter reconciliation function.

        It applies the Dual-LPHM parameter reconciliation operation to the input parameter vector $\mathbf{w}$,
        and returns the reconciled parameter matrix of shape (n, D) subject to the dimension and rank parameters
        $p$, $q$ and $r$ as follows:
        $$
            \begin{equation}
                \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = ( \mathbf{P} \mathbf{Q}^\top) \otimes ( \mathbf{S} \mathbf{T}^\top) \in {R}^{n \times D}.
            \end{equation}
        $$
        where $\mathbf{P} \in {R}^{p \times r}$, $\mathbf{Q} \in {R}^{q \times r}$, $\mathbf{S} \in {R}^{\frac{n}{p} \times r}$ and
        $\mathbf{T} \in {R}^{\frac{D}{q} \times r}$ are all obtained by partitioning $\mathbf{w}$ into sub-vectors
        and subsequently reshaping them into matrices.

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
        s, t = int(n/self.p), int(D/self.q)
        P, Q, S, T = torch.split(w, [self.p*self.r, self.q*self.r, s*self.r, t*self.r], dim=1)
        A = F.linear(P.view(self.p, -1), Q.view(self.q, -1)).view(1, -1)
        B = F.linear(S.view(s, -1), T.view(t, -1)).view(1, -1)
        return torch.einsum('pq,st->psqt', A, B).view(self.p*s, self.q*t)