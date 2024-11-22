# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################
# Parameterized Interdependence #
#################################
"""
The parameterized interdependence functions

This module contains the parameterized interdependence functions, including
    parameterized_interdependence,
    lowrank_parameterized_interdependence,
    hm_parameterized_interdependence,
    lphm_parameterized_interdependence,
    dual_lphm_parameterized_interdependence,
    random_matrix_adaption_parameterized_interdependence
"""

import torch

from tinybig.interdependence import interdependence
from tinybig.reconciliation import (
    lorr_reconciliation,
    hm_reconciliation,
    lphm_reconciliation,
    dual_lphm_reconciliation,
    random_matrix_adaption_reconciliation
)


class parameterized_interdependence(interdependence):
    r"""
        A parameterized interdependence function.

        This class allows the computation of interdependence matrices parameterized by custom
        fabrication methods or predefined structures.

        Notes
        ----------
        Formally, given a learnable parameter vector $\mathbf{w} \in R^{l_{\xi}}$, the parameterized interdependence function transforms it into a matrix of desired dimensions $m \times m'$ as follows:

        $$
            \begin{equation}
            \xi(\mathbf{w}) = \text{reshape}(\mathbf{w}) = \mathbf{W} \in R^{m \times m'}.
            \end{equation}
        $$

        This parameterized interdependence function operates independently of any data batch, deriving the output interdependence matrix solely from the learnable parameter vector $\mathbf{w}$,
        whose required length of vector $\mathbf{w}$ is $l_{\xi} = m \times m'$.

        Attributes
        ----------
        parameter_fabrication : Callable
            A callable function or object to fabricate parameters.
        b_prime : int
            The number of rows in the output interdependence matrix.
        m_prime : int
            The number of columns in the output interdependence matrix.

        Methods
        -------
        calculate_l()
            Computes the total number of parameters needed.
        calculate_b_prime(b=None)
            Computes the effective number of rows in the interdependence matrix.
        calculate_m_prime(m=None)
            Computes the effective number of columns in the interdependence matrix.
        calculate_A(x=None, w=None, device='cpu', *args, **kwargs)
            Computes the parameterized interdependence matrix.
    """
    def __init__(
        self,
        b: int, m: int,
        b_prime: int = None, m_prime: int = None,
        interdependence_type: str = 'attribute',
        name: str = 'parameterized_interdependence',
        require_parameters: bool = True,
        require_data: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        """
            Initializes the parameterized interdependence function.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            b_prime : int, optional
                Number of rows in the output interdependence matrix. Defaults to `b`.
            m_prime : int, optional
                Number of columns in the output interdependence matrix. Defaults to `m`.
            interdependence_type : str, optional
                Type of interdependence ('instance', 'attribute', etc.). Defaults to 'attribute'.
            name : str, optional
                Name of the interdependence function. Defaults to 'parameterized_interdependence'.
            require_parameters : bool, optional
                Whether parameters are required. Defaults to True.
            require_data : bool, optional
                Whether input data is required. Defaults to False.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data, require_parameters=require_parameters, device=device, *args, **kwargs)
        self.parameter_fabrication = None
        self.b_prime = b_prime if b_prime is not None else b
        self.m_prime = m_prime if m_prime is not None else m

    def calculate_l(self):
        """
            Computes the total number of parameters required.

            Returns
            -------
            int
                The total number of parameters.

            Raises
            ------
            ValueError
                If the interdependence type is not supported.
        """
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

        assert d is not None and d_prime is not None
        if self.parameter_fabrication is None:
            return d * d_prime
        else:
            return self.parameter_fabrication.calculate_l(n=d, D=d_prime)

    def calculate_b_prime(self, b: int = None):
        """
            Computes the effective number of rows in the interdependence matrix.

            Parameters
            ----------
            b : int, optional
                Input number of rows. Defaults to None.

            Returns
            -------
            int
                The effective number of rows in the matrix.
        """
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence'] and self.b_prime is not None:
            return self.b_prime
        else:
            return b if b is not None else self.b

    def calculate_m_prime(self, m: int = None):
        """
            Computes the effective number of columns in the interdependence matrix.

            Parameters
            ----------
            m : int, optional
                Input number of columns. Defaults to None.

            Returns
            -------
            int
                The effective number of columns in the matrix.
        """
        if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence'] and self.m_prime is not None:
            return self.m_prime
        else:
            return m if m is not None else self.m

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the parameterized interdependence matrix.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor of shape `(num_parameters,)`. Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The computed interdependence matrix.

            Raises
            ------
            ValueError
                If the interdependence type is not supported.
            AssertionError
                If the parameter tensor `w` has an incorrect shape.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            assert w.ndim == 2 and w.numel() == self.calculate_l()

            if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                d, d_prime = self.b, self.calculate_b_prime()
            elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                d, d_prime = self.m, self.calculate_m_prime()
            else:
                raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

            if self.parameter_fabrication is None:
                A = w.reshape(d, d_prime).to(device=device)
            else:
                A = self.parameter_fabrication(w=w, n=d, D=d_prime, device=device)
            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A


class lowrank_parameterized_interdependence(parameterized_interdependence):
    r"""
        A low-rank parameterized interdependence function.

        Notes
        ----------

        Formally, given the parameter vector $\mathbf{w} \in R^{l_{\xi}}$ and a rank hyper-parameter $r$,
        we partition $\mathbf{w}$ into two sub-vectors and subsequently reshape them into two matrices
        $\mathbf{A} \in R^{m \times r}$ and $\mathbf{B} \in R^{m' \times r}$,
        each possessing a rank of $r$.

        These two sub-matrices $\mathbf{A}$ and $\mathbf{B}$ help define the low-rank parameterized interdependence function as follows:

        $$
            \begin{equation}
            \xi(\mathbf{w}) = \mathbf{A} \mathbf{B}^\top \in R^{m \times m'},
            \end{equation}
        $$
        whose required length of vector $\mathbf{w}$ is $l_{\xi} = (m + m') \times r$.

        Attributes
        ----------
        r : int
            Rank of the low-rank approximation.

        Methods
        -------
        __init__(...)
            Initializes the low-rank parameterized interdependence function.
    """
    def __init__(self, r: int = 2, name: str = 'lowrank_parameterized_interdependence', *args, **kwargs):
        """
            Initializes the low-rank parameterized interdependence function.

            Parameters
            ----------
            r : int, optional
                Rank of the low-rank approximation. Defaults to 2.
            name : str, optional
                Name of the interdependence function. Defaults to 'lowrank_parameterized_interdependence'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = lorr_reconciliation(r=self.r)


class hm_parameterized_interdependence(parameterized_interdependence):
    r"""
        A parameterized interdependence function using hierarchical mapping (HM).

        Notes
        ----------

        Formally, given the parameter vector $\mathbf{w} \in R^{l_{\xi}}$, we partition $\mathbf{w}$ into two sub-vectors and subsequently reshape them into two matrices
        $\mathbf{A} \in R^{p \times q}$ and $\mathbf{B} \in R^{s \times t}$ (where $s =\frac{m}{p}$ and $t = \frac{m'}{q}$).

        These two sub-matrices $\mathbf{A}$ and $\mathbf{B}$ help define the hypercomplex parameterized interdependence function as follows:

        $$
            \begin{equation}
            \xi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} \in R^{m \times m'},
            \end{equation}
        $$
        whose required length of vector $\mathbf{w}$ is $l_{\xi} = pq + \frac{mm'}{pq}$.

        Attributes
        ----------
        p : int
            Number of partitions in the input dimension.
        q : int
            Number of partitions in the output dimension.

        Methods
        -------
        __init__(...)
            Initializes the hierarchical mapping parameterized interdependence function.
    """
    def __init__(self, p: int, q: int = None, name: str = 'hm_parameterized_interdependence', *args, **kwargs):
        """
            Initializes the hierarchical mapping parameterized interdependence function.

            Parameters
            ----------
            p : int
                Number of partitions in the input dimension.
            q : int, optional
                Number of partitions in the output dimension. Defaults to `p`.
            name : str, optional
                Name of the interdependence function. Defaults to 'hm_parameterized_interdependence'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Raises
            ------
            ValueError
                If the interdependence type is not supported.
            AssertionError
                If the dimensions are not divisible by the partitions.
        """
        super().__init__(name=name, *args, **kwargs)

        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = hm_reconciliation(p=self.p, q=self.q)


class lphm_parameterized_interdependence(parameterized_interdependence):
    r"""
        A parameterized interdependence function using low-rank hierarchical mapping (LPHM).

        Notes
        ----------

        Formally, given the parameter vector $\mathbf{w} \in R^{l_{\xi}}$ and rank hyper-parameter $r$, we partition $\mathbf{w}$ into three sub-vectors and subsequently reshape them into three matrices
        $\mathbf{A} \in R^{p \times q}$, $\mathbf{S} \in R^{\frac{m}{p} \times r}$ and $\mathbf{T} \in R^{\frac{m'}{q} \times r}$.

        These three sub-matrices $\mathbf{A}$, $\mathbf{S}$ and $\mathbf{T}$ help define the low-rank hypercomplex parameterized interdependence function as follows:

        $$
            \begin{equation}
            \xi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = \mathbf{A} \otimes ( \mathbf{S} \mathbf{T}^\top) \in R^{m \times m'},
            \end{equation}
        $$
        whose required length of vector $\mathbf{w}$ is $l_{\xi} = pq + r \times (\frac{m}{p} + \frac{m'}{q})$.

        Attributes
        ----------
        r : int
            Rank of the low-rank approximation.
        p : int
            Number of partitions in the input dimension.
        q : int
            Number of partitions in the output dimension.

        Methods
        -------
        __init__(...)
            Initializes the low-rank hierarchical mapping parameterized interdependence function.
    """
    def __init__(self, r: int, p: int, q: int = None, name: str = 'lphm_parameterized_interdependence', *args, **kwargs):
        """
            Initializes the low-rank hierarchical mapping (LPHM) parameterized interdependence function.

            Parameters
            ----------
            r : int
                Rank of the low-rank approximation.
            p : int
                Number of partitions in the input dimension.
            q : int, optional
                Number of partitions in the output dimension. Defaults to `p`.
            name : str, optional
                Name of the interdependence function. Defaults to 'lphm_parameterized_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            ValueError
                If the interdependence type is not supported.
            AssertionError
                If the input and output dimensions are not divisible by their respective partitions.
        """
        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class dual_lphm_parameterized_interdependence(parameterized_interdependence):
    r"""
        A parameterized interdependence function using dual low-rank hierarchical mapping.

        Notes
        ----------

        Formally, given the parameter vector $\mathbf{w} \in R^{l_{\xi}}$ and rank hyper-parameter $r$, we partition $\mathbf{w}$ into three sub-vectors and subsequently reshape them into four matrices
        $\mathbf{P} \in R^{p \times r}$, $\mathbf{Q} \in R^{q \times r}$,  $\mathbf{S} \in R^{\frac{m}{p} \times r}$ and $\mathbf{T} \in R^{\frac{m'}{q} \times r}$.

        These four sub-matrices $\mathbf{A}$, $\mathbf{S}$ and $\mathbf{T}$ help define the dual low-rank hypercomplex parameterized interdependence function as follows:

        $$
            \begin{equation}
            \xi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = ( \mathbf{P} \mathbf{Q}^\top) \otimes ( \mathbf{S} \mathbf{T}^\top) \in R^{m \times m'},
            \end{equation}
        $$
        whose required length of vector $\mathbf{w}$ is $l_{\xi} = r \times (p + q + \frac{m}{p} + \frac{m'}{q})$.

        Attributes
        ----------
        r : int
            Rank of the low-rank approximation.
        p : int
            Number of partitions in the input dimension.
        q : int
            Number of partitions in the output dimension.

        Methods
        -------
        __init__(...)
            Initializes the dual low-rank hierarchical mapping parameterized interdependence function.
    """
    def __init__(self, r: int, p: int, q: int = None, name: str = 'dual_lphm_parameterized_interdependence', *args, **kwargs):
        """
            Initializes the dual low-rank hierarchical mapping (Dual-LPHM) parameterized interdependence function.

            Parameters
            ----------
            r : int
                Rank of the low-rank approximation.
            p : int
                Number of partitions in the input dimension.
            q : int, optional
                Number of partitions in the output dimension. Defaults to `p`.
            name : str, optional
                Name of the interdependence function. Defaults to 'dual_lphm_parameterized_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            ValueError
                If the interdependence type is not supported.
            AssertionError
                If the input and output dimensions are not divisible by their respective partitions.
        """
        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = dual_lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class random_matrix_adaption_parameterized_interdependence(parameterized_interdependence):
    r"""
        A parameterized interdependence function using random matrix adaptation.

        Notes
        ----------

        Formally, given a parameter vector $\mathbf{w} \in R^l$ of length $l$, we can partition it into two vectors $\lambda_1$ and $\lambda_2$.
        These two vectors will define two diagonal matrices $\Lambda_1 = diag( \lambda_1) \in R^{m \times m}$ and $\Lambda_2 = diag(\lambda_2) \in R^{r \times r}$.

        These two sub-matrices will fabricate a parameter matrix of shape $m \times m'$ as follows:

        $$
            \begin{equation}
            \xi(\mathbf{w}) =  \Lambda_1 \mathbf{A} \Lambda_1 \mathbf{B}^\top \in R^{m \times m'},
            \end{equation}
        $$

        where matrices $\mathbf{A} \in R^{m \times r}$ and $\mathbf{B} \in R^{r \times m'}$ are randomly sampled from the Gaussian distribution $\mathcal{N}(\mathbf{0}, \mathbf{I})$.
        The required length of vector $\mathbf{w}$ is $l_{\xi} = m + r$.


        Attributes
        ----------
        r : int
            Rank of the random matrix approximation.

        Methods
        -------
        __init__(...)
            Initializes the random matrix adaptation parameterized interdependence function.
    """
    def __init__(self, r: int = 2, name: str = 'random_matrix_adaption_parameterized_interdependence', *args, **kwargs):
        """
            Initializes the random matrix adaptation parameterized interdependence function.

            Parameters
            ----------
            r : int, optional
                Rank of the random matrix approximation. Defaults to 2.
            name : str, optional
                Name of the interdependence function. Defaults to 'random_matrix_adaption_parameterized_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = random_matrix_adaption_reconciliation(r=self.r)
