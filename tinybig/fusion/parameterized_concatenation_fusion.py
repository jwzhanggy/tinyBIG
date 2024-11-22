# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

######################################################
# Parameterized Concatenation based Fusion Functions #
######################################################

"""
The parameterized concatenation based fusion functions

This module contains the parameterized concatenation based fusion function, such as
    parameterized_concatenation_fusion,
    lowrank_parameterized_concatenation_fusion,
    hm_parameterized_concatenation_fusion,
    lphm_parameterized_concatenation_fusion,
    dual_lphm_parameterized_concatenation_fusion,
    random_matrix_adaption_parameterized_concatenation_fusion
"""

import torch

from tinybig.fusion import fusion
from tinybig.reconciliation import (
    lorr_reconciliation,
    hm_reconciliation,
    lphm_reconciliation,
    dual_lphm_reconciliation,
    random_matrix_adaption_reconciliation
)


class parameterized_concatenation_fusion(fusion):
    r"""
        A fusion mechanism that concatenates input tensors along their last dimension, followed by a learnable parameterized transformation.

        Notes
        ----------

        Formally, given input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k$,
        where each matrix $\mathbf{A}_i \in R^{m \times n_i}$ has $m$ rows and $n_i$ columns,
        we define the fusion operator as follows:

        $$
            \begin{equation}
            \begin{aligned}
            \mathbf{A} &= \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) \\
            &= \left( \mathbf{A}_1 \sqcup \mathbf{A}_2 \sqcup \cdots \sqcup \mathbf{A}_k \right) \mathbf{W} \in R^{m \times n},
            \end{aligned}
            \end{equation}
        $$

        where $\sqcup$ denotes the row-wise concatenation of the matrices. The term $\mathbf{W} \in R^{(\sum_{i=1}^k n_i) \times n}$
        represents a learnable parameter matrix that projects the concatenated matrix to a dimension of $n$.

        The number of required learnable parameter for this fusion function will be $l = (\sum_{i=1}^k n_i) \times n$.

        Attributes
        ----------
        n : int
            Output dimension after transformation.
        dims : list[int] | tuple[int]
            List or tuple specifying the dimensions of the input tensors.
        parameter_fabrication : Callable
            Function or object to handle parameter generation or transformation.

        Methods
        -------
        calculate_n(dims=None, *args, **kwargs)
            Computes the output dimension after the parameterized transformation.
        calculate_l(*args, **kwargs)
            Computes the number of learnable parameters for the transformation.
        forward(x, w=None, device='cpu', *args, **kwargs)
            Performs the concatenation fusion followed by the parameterized transformation.
    """

    def __init__(self, n: int = None, dims: list[int] | tuple[int] = None, name: str = "parameterized_concatenation_fusion", require_parameters: bool = True, *args, **kwargs):
        """
            Initializes the parameterized concatenation fusion function.

            Parameters
            ----------
            n : int, optional
                Output dimension after transformation. Defaults to None.
            dims : list[int] | tuple[int], optional
                List or tuple specifying the dimensions of the input tensors. Defaults to None.
            name : str, optional
                Name of the fusion function. Defaults to "parameterized_concatenation_fusion".
            require_parameters : bool, optional
                Indicates whether the fusion requires learnable parameters. Defaults to True.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(dims=dims, name=name, require_parameters=True, *args, **kwargs)
        if n is not None:
            self.n = n
        else:
            assert dims is not None and all([dim == dims[0] for dim in dims])
            self.n = dims[0]
        self.parameter_fabrication = None

    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        """
            Computes the output dimension after the parameterized transformation.

            Parameters
            ----------
            dims : list[int] | tuple[int], optional
                List or tuple specifying the dimensions of the input tensors. Defaults to None.

            Returns
            -------
            int
                Output dimension after transformation.

            Raises
            ------
            AssertionError
                If `dims` is inconsistent or not provided.
        """
        if self.n is not None:
            return self.n
        else:
            dims = dims if dims is not None else self.dims
            assert dims is not None and all([dim == dims[0] for dim in dims])
            return dims[0]

    def calculate_l(self, *args, **kwargs):
        """
            Computes the number of learnable parameters for the transformation.

            Returns
            -------
            int
                Total number of learnable parameters.

            Raises
            ------
            ValueError
                If `dims` or `n` is not specified.
        """
        if self.dims is None or self.n is None:
            raise ValueError("The output dimension n is required...")
        if self.parameter_fabrication is None:
            return sum(self.dims) * self.n
        else:
            return self.parameter_fabrication.calculate_l(n=self.n, D=sum(self.dims))

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor], w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Performs the concatenation fusion followed by the parameterized transformation.

            Parameters
            ----------
            x : list[torch.Tensor] | tuple[torch.Tensor]
                List or tuple of input tensors to be concatenated and transformed.
            w : torch.nn.Parameter, optional
                Learnable weights for the transformation. Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Fused and transformed tensor.

            Raises
            ------
            ValueError
                If `x` is empty or if `dims` or `n` is not specified.
        """
        if not x:
            raise ValueError("The input x cannot be empty...")
        if not all(x[0].shape[:-1] == t.shape[:-1] for t in x):
            raise ValueError("Excluding the last dimension, the input x contains elements of different shapes for other dimensions...")

        if all(x[0].shape == t.shape for t in x):
            # if they are all the same shape, it will allow some cross-channel pre-processing operators...
            x = torch.stack(x, dim=0)
            x = self.pre_process(x=x, device=device)
            x = [t.squeeze(dim=0) for t in x.split(1, dim=0)]
        else:
            # otherwise, we cannot perform cross channel preprocessing, and have to pre-process them individually...
            x = [self.pre_process(t, device=device) for t in x]

        x = torch.cat(x, dim=-1)

        if self.dims is None or self.n is None:
            raise ValueError("The output dimension n is required...")
        if self.parameter_fabrication is None:
            W = w.reshape(self.n, sum(self.dims)).to(device=device)
        else:
            W = self.parameter_fabrication(w=w, n=self.n, D=sum(self.dims), device=device)

        fused_x = torch.matmul(x, W.t())

        assert fused_x.size(-1) == self.calculate_n([element.size(-1) for element in x])
        return self.post_process(x=fused_x, device=device)


class lowrank_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    r"""
        A parameterized concatenation fusion with a low-rank approximation for parameter fabrication.

        Notes
        ----------

        Formally, given input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k$,
        where each matrix $\mathbf{A}_i \in R^{m \times n_i}$ has $m$ rows and $n_i$ columns,
        we define the fusion operator as follows:

        $$
            \begin{equation}
            \begin{aligned}
            \mathbf{A} &= \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) \\
            &= \left( \mathbf{A}_1 \sqcup \mathbf{A}_2 \sqcup \cdots \sqcup \mathbf{A}_k \right) \mathbf{W} \in R^{m \times n},
            \end{aligned}
            \end{equation}
        $$

        where $\sqcup$ denotes the row-wise concatenation of the matrices.

        Notation $\mathbf{W} \in R^{(\sum_{i=1}^k n_i) \times n}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \mathbf{B}^\top = \mathbf{W} \in R^{(\sum_{i=1}^k n_i) \times n},
            \end{equation}
        $$
        where $\mathbf{A} \in R^{(\sum_{i=1}^k n_i) \times r}$ and $\mathbf{B} \in R^{n \times r}$ are partitioned and reshaped from the parameter vector $\mathbf{w}$.

        The required length of parameter vector of this interdependence function is $l = ((\sum_{i=1}^k n_i) + n) \times r$.



        Attributes
        ----------
        r : int
            Rank of the low-rank approximation.

        Methods
        -------
        __init__(...)
            Initializes the low-rank parameterized concatenation fusion function.
    """
    def __init__(self, r: int = 2, name: str = 'lowrank_parameterized_concatenation_fusion', *args, **kwargs):
        """
            Initializes the low-rank parameterized concatenation fusion function.

            Parameters
            ----------
            r : int, optional
                Rank of the low-rank approximation. Defaults to 2.
            name : str, optional
                Name of the fusion function. Defaults to "lowrank_parameterized_concatenation_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = lorr_reconciliation(r=self.r)


class hm_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    r"""
        A parameterized concatenation fusion with hierarchical matrix (HM) parameter fabrication.

        Notes
        ----------

        Formally, given input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k$,
        where each matrix $\mathbf{A}_i \in R^{m \times n_i}$ has $m$ rows and $n_i$ columns,
        we define the fusion operator as follows:

        $$
            \begin{equation}
            \begin{aligned}
            \mathbf{A} &= \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) \\
            &= \left( \mathbf{A}_1 \sqcup \mathbf{A}_2 \sqcup \cdots \sqcup \mathbf{A}_k \right) \mathbf{W} \in R^{m \times n},
            \end{aligned}
            \end{equation}
        $$

        where $\sqcup$ denotes the row-wise concatenation of the matrices.

        Notation $\mathbf{W} \in R^{(\sum_{i=1}^k n_i) \times n}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} \in R^{(\sum_{i=1}^k n_i) \times n},
            \end{equation}
        $$
        where $\mathbf{A} \in R^{p \times q}$ and $\mathbf{B} \in R^{s \times t}$ (where $s =\frac{(\sum_{i=1}^k n_i)}{p}$ and $t = \frac{n}{q}$) are partitioned and reshaped from the parameter vector $\mathbf{w}$.

        The required length of parameter vector of this interdependence function is $l = pq + \frac{(\sum_{i=1}^k n_i)n}{pq}$.

        Attributes
        ----------
        p : int
            Partition size for the hierarchical matrix.
        q : int
            Block size for the hierarchical matrix.

        Methods
        -------
        __init__(...)
            Initializes the HM parameterized concatenation fusion function.
    """
    def __init__(self, p: int, q: int = None, name: str = 'hm_parameterized_concatenation_fusion', *args, **kwargs):
        """
            Initializes the HM parameterized concatenation fusion function.

            Parameters
            ----------
            p : int
                Partition size for the hierarchical matrix.
            q : int, optional
                Block size for the hierarchical matrix. Defaults to `p`.
            name : str, optional
                Name of the fusion function. Defaults to "hm_parameterized_concatenation_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            AssertionError
                If `n` is not divisible by `p` or if the sum of `dims` is not divisible by `q`.
        """
        super().__init__(name=name, *args, **kwargs)
        self.p = p
        self.q = q if q is not None else p
        assert self.n is not None and self.n % self.p == 0
        assert self.dims is not None and sum(self.dims) % self.q == 0
        self.parameter_fabrication = hm_reconciliation(p=self.p, q=self.q)


class lphm_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    r"""
        A parameterized concatenation fusion with low-rank hierarchical matrix (LPHM) parameter fabrication.

        Notes
        ----------

        Formally, given input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k$,
        where each matrix $\mathbf{A}_i \in R^{m \times n_i}$ has $m$ rows and $n_i$ columns,
        we define the fusion operator as follows:

        $$
            \begin{equation}
            \begin{aligned}
            \mathbf{A} &= \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) \\
            &= \left( \mathbf{A}_1 \sqcup \mathbf{A}_2 \sqcup \cdots \sqcup \mathbf{A}_k \right) \mathbf{W} \in R^{m \times n},
            \end{aligned}
            \end{equation}
        $$

        where $\sqcup$ denotes the row-wise concatenation of the matrices.

        Notation $\mathbf{W} \in R^{(\sum_{i=1}^k n_i) \times n}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = \mathbf{A} \otimes ( \mathbf{S} \mathbf{T}^\top) \in R^{(\sum_{i=1}^k n_i) \times n},
            \end{equation}
        $$
        where $\mathbf{A} \in R^{p \times q}$, $\mathbf{S} \in R^{\frac{(\sum_{i=1}^k n_i)}{p} \times r}$ and $\mathbf{T} \in R^{\frac{n}{q} \times r}$ are partitioned and reshaped from the parameter vector $\mathbf{w}$.

        The required length of parameter vector of this interdependence function is $l = pq + r \times (\frac{(\sum_{i=1}^k n_i)}{p} + \frac{n}{q})$.


        Attributes
        ----------
        r : int
            Rank for the low-rank approximation.
        p : int
            Partition size for the hierarchical matrix.
        q : int
            Block size for the hierarchical matrix.

        Methods
        -------
        __init__(...)
            Initializes the LPHM parameterized concatenation fusion function.
    """
    def __init__(self, r: int, p: int, q: int = None, name: str = 'lphm_parameterized_concatenation_fusion', *args, **kwargs):
        """
            Initializes the LPHM parameterized concatenation fusion function.

            Parameters
            ----------
            r : int
                Rank for the low-rank approximation.
            p : int
                Partition size for the hierarchical matrix.
            q : int, optional
                Block size for the hierarchical matrix. Defaults to `p`.
            name : str, optional
                Name of the fusion function. Defaults to "lphm_parameterized_concatenation_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            AssertionError
                If `n` is not divisible by `p` or if the sum of `dims` is not divisible by `q`.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.p = p
        self.q = q if q is not None else p
        assert self.n is not None and self.n % self.p == 0
        assert self.dims is not None and sum(self.dims) % self.q == 0
        self.parameter_fabrication = lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class dual_lphm_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    r"""
        A parameterized concatenation fusion with dual low-rank hierarchical matrix (Dual-LPHM) parameter fabrication.

        Notes
        ----------

        Formally, given input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k$,
        where each matrix $\mathbf{A}_i \in R^{m \times n_i}$ has $m$ rows and $n_i$ columns,
        we define the fusion operator as follows:

        $$
            \begin{equation}
            \begin{aligned}
            \mathbf{A} &= \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) \\
            &= \left( \mathbf{A}_1 \sqcup \mathbf{A}_2 \sqcup \cdots \sqcup \mathbf{A}_k \right) \mathbf{W} \in R^{m \times n},
            \end{aligned}
            \end{equation}
        $$

        where $\sqcup$ denotes the row-wise concatenation of the matrices.

        Notation $\mathbf{W} \in R^{(\sum_{i=1}^k n_i) \times n}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = ( \mathbf{P} \mathbf{Q}^\top) \otimes ( \mathbf{S} \mathbf{T}^\top) \in R^{(\sum_{i=1}^k n_i) \times n},
            \end{equation}
        $$
        where $\mathbf{P} \in R^{p \times r}$, $\mathbf{Q} \in R^{q \times r}$,  $\mathbf{S} \in R^{\frac{(\sum_{i=1}^k n_i)}{p} \times r}$ and $\mathbf{T} \in R^{\frac{n}{q} \times r}$ are partitioned and reshaped from the parameter vector $\mathbf{w}$.

        The required length of parameter vector of this interdependence function is $l = r \times (p + q + \frac{(\sum_{i=1}^k n_i)}{p} + \frac{n}{q})$.

        Attributes
        ----------
        r : int
            Rank for the low-rank approximation.
        p : int
            Partition size for the hierarchical matrix.
        q : int
            Block size for the hierarchical matrix.

        Methods
        -------
        __init__(...)
            Initializes the Dual-LPHM parameterized concatenation fusion function.
    """
    def __init__(self, r: int, p: int, q: int = None, name: str = 'dual_lphm_parameterized_concatenation_fusion', *args, **kwargs):
        """
            Initializes the Dual-LPHM parameterized concatenation fusion function.

            Parameters
            ----------
            r : int
                Rank for the low-rank approximation.
            p : int
                Partition size for the hierarchical matrix.
            q : int, optional
                Block size for the hierarchical matrix. Defaults to `p`.
            name : str, optional
                Name of the fusion function. Defaults to "dual_lphm_parameterized_concatenation_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            AssertionError
                If `n` is not divisible by `p` or if the sum of `dims` is not divisible by `q`.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.p = p
        self.q = q if q is not None else p
        assert self.n is not None and self.n % self.p == 0
        assert self.dims is not None and sum(self.dims) % self.q == 0
        self.parameter_fabrication = dual_lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class random_matrix_adaption_parameterized_concatenation_fusion(parameterized_concatenation_fusion):
    r"""
        A parameterized concatenation fusion with random matrix adaptation for parameter fabrication.

        Notes
        ----------

        Formally, given input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k$,
        where each matrix $\mathbf{A}_i \in R^{m \times n_i}$ has $m$ rows and $n_i$ columns,
        we define the fusion operator as follows:

        $$
            \begin{equation}
            \begin{aligned}
            \mathbf{A} &= \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) \\
            &= \left( \mathbf{A}_1 \sqcup \mathbf{A}_2 \sqcup \cdots \sqcup \mathbf{A}_k \right) \mathbf{W} \in R^{m \times n},
            \end{aligned}
            \end{equation}
        $$

        where $\sqcup$ denotes the row-wise concatenation of the matrices.

        Notation $\mathbf{W} \in R^{(\sum_{i=1}^k n_i) \times n}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \Lambda_1 \mathbf{A} \Lambda_1 \mathbf{B}^\top \in R^{(\sum_{i=1}^k n_i) \times n},
            \end{equation}
        $$

        Notations $\Lambda_1$ and $\Lambda_2$ denote two diagonal matrices $\Lambda_1 = diag( \lambda_1) \in R^{(\sum_{i=1}^k n_i) \times (\sum_{i=1}^k n_i)}$ and $\Lambda_2 = diag(\lambda_2) \in R^{r \times r}$
        where the diagonal vectors $\lambda_1$ and $\lambda_2$ are partitioned from the parameter vector $\mathbf{w}$.
        Matrices $\mathbf{A} \in R^{(\sum_{i=1}^k n_i) \times r}$ and $\mathbf{B} \in R^{r \times n}$ are randomly sampled from the Gaussian distribution $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

        The required length of parameter vector of this interdependence function is $\mathbf{w}$ is $l = (\sum_{i=1}^k n_i) + r$.



        Attributes
        ----------
        r : int
            Rank for the random matrix adaptation.

        Methods
        -------
        __init__(...)
            Initializes the random matrix adaptation parameterized concatenation fusion function.
    """
    def __init__(self, r: int = 2, name: str = 'random_matrix_adaption_parameterized_concatenation_fusion', *args, **kwargs):
        """
            Initializes the random matrix adaptation parameterized concatenation fusion function.

            Parameters
            ----------
            r : int, optional
                Rank for the random matrix adaptation. Defaults to 2.
            name : str, optional
                Name of the fusion function. Defaults to "random_matrix_adaption_parameterized_concatenation_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = random_matrix_adaption_reconciliation(r=self.r)
