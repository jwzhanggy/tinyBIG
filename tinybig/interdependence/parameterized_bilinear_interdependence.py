# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################################
# Parameterized Bilinear Interdependence #
##########################################
"""
The parameterized bilinear interdependence functions

This module contains the parameterized bilinear interdependence functions, including
    parameterized_bilinear_interdependence,
    lowrank_parameterized_bilinear_interdependence,
    hm_parameterized_bilinear_interdependence,
    lphm_parameterized_bilinear_interdependence,
    dual_lphm_parameterized_bilinear_interdependence,
    random_matrix_adaption_parameterized_bilinear_interdependence
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


class parameterized_bilinear_interdependence(interdependence):
    r"""
        A parameterized bilinear interdependence function.

        This class computes interdependence matrices using a bilinear transformation
        parameterized by custom fabrication methods or predefined structures.

        Notes
        ----------
        Formally, given a data batch $\mathbf{X} \in R^{b \times m}$, we can represent the parameterized bilinear form-based interdependence function as follows:

        $$
            \begin{equation}\label{equ:bilinear_interdependence_function}
            \xi(\mathbf{X} | \mathbf{w}) = \mathbf{X}^\top \mathbf{W} \mathbf{X} = \mathbf{A} \in R^{m \times m},
            \end{equation}
        $$

        where $\mathbf{W} = \text{reshape}(\mathbf{w}) \in R^{b \times b}$ denotes the parameter matrix reshaped from the learnable parameter vector $\mathbf{w} \in R^{l_{\xi}}$.

        The required length of parameter vector of this interdependence function is $l_{\xi} = b^2$.


        Attributes
        ----------
        parameter_fabrication : Callable
            A callable function or object to fabricate parameters.

        Methods
        -------
        calculate_b_prime(b=None)
            Computes the effective number of rows in the interdependence matrix.
        calculate_m_prime(m=None)
            Computes the effective number of columns in the interdependence matrix.
        calculate_l()
            Computes the total number of parameters needed.
        calculate_A(x=None, w=None, device='cpu', *args, **kwargs)
            Computes the parameterized bilinear interdependence matrix.
    """

    def __init__(
        self,
        b: int, m: int,
        interdependence_type: str = 'instance',
        name: str = 'parameterized_bilinear_interdependence',
        require_parameters: bool = True,
        require_data: bool = True,
        device: str = 'cpu', *args, **kwargs
    ):
        """
            Initializes the parameterized bilinear interdependence function.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            interdependence_type : str, optional
                Type of interdependence ('instance', 'attribute', etc.). Defaults to 'instance'.
            name : str, optional
                Name of the interdependence function. Defaults to 'parameterized_bilinear_interdependence'.
            require_parameters : bool, optional
                Whether parameters are required. Defaults to True.
            require_data : bool, optional
                Whether input data is required. Defaults to True.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data, require_parameters=require_parameters, device=device, *args, **kwargs)
        self.parameter_fabrication = None

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
        b = b if b is not None else self.b
        return b

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
        m = m if m is not None else self.m
        return m

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
            if self.parameter_fabrication is None:
                return self.m ** 2
            else:
                return self.parameter_fabrication.calculate_l(n=self.m, D=self.m)
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            if self.parameter_fabrication is None:
                return self.b ** 2
            else:
                return self.parameter_fabrication.calculate_l(n=self.b, D=self.b)
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the parameterized bilinear interdependence matrix.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Required for computation.
            w : torch.nn.Parameter, optional
                Parameter tensor of shape `(num_parameters,)`. Required for computation.
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
                If input data or parameter tensor `w` has an incorrect shape.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            assert x is not None and x.ndim == 2
            assert w is not None and w.ndim == 2 and w.numel() == self.calculate_l()

            x = self.pre_process(x=x, device=device)

            if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                # for instance interdependence, the parameter for calculating x.t*W*x will have dimension m*m'
                d, d_prime = self.m, self.calculate_m_prime()
            elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                # for attribute interdependence, the parameter for calculating x.t*W*x will have dimension b*b'
                d, d_prime = self.b, self.calculate_b_prime()
            else:
                raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

            if self.parameter_fabrication is None:
                W = w.reshape(d, d_prime).to(device=device)
            else:
                W = self.parameter_fabrication(w=w, n=d, D=d_prime, device=device)

            A = torch.matmul(x.t(), torch.matmul(W, x))
            A = self.post_process(x=A, device=device)

            # if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            #     assert A.shape == (self.m, self.calculate_m_prime())
            # elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            #     assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A


class lowrank_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    r"""
        A low-rank parameterized bilinear interdependence function.

        Notes
        ----------
        Formally, given a data batch $\mathbf{X} \in R^{b \times m}$, we can represent the parameterized bilinear form-based interdependence function as follows:

        $$
            \begin{equation}\label{equ:bilinear_interdependence_function}
            \xi(\mathbf{X} | \mathbf{w}) = \mathbf{X}^\top \mathbf{W} \mathbf{X} = \mathbf{A} \in R^{m \times m}.
            \end{equation}
        $$

        Notation $\mathbf{W} \in R^{b \times b}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l_{\xi}}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \mathbf{B}^\top \in R^{b \times b},
            \end{equation}
        $$
        where $\mathbf{A} \in R^{b \times r}$ and $\mathbf{B} \in R^{b \times r}$ are partitioned and reshaped from the parameter vector $\mathbf{w}$.

        The required length of parameter vector of this interdependence function is $l_{\xi} = (b + b) \times r$.

        Attributes
        ----------
        r : int
            Rank of the low-rank approximation.

        Methods
        -------
        __init__(...)
            Initializes the low-rank parameterized bilinear interdependence function.
    """
    def __init__(self, r: int = 2, name: str = 'lowrank_parameterized_bilinear_interdependence', *args, **kwargs):
        """
            Initializes the low-rank parameterized bilinear interdependence function.

            Parameters
            ----------
            r : int, optional
                Rank of the low-rank approximation. Defaults to 2.
            name : str, optional
                Name of the interdependence function. Defaults to 'lowrank_parameterized_bilinear_interdependence'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = lorr_reconciliation(r=self.r)


class hm_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    r"""
        A hierarchical mapping (HM) parameterized bilinear interdependence function.

        Notes
        ----------
        Formally, given a data batch $\mathbf{X} \in R^{b \times m}$, we can represent the parameterized bilinear form-based interdependence function as follows:

        $$
            \begin{equation}\label{equ:bilinear_interdependence_function}
            \xi(\mathbf{X} | \mathbf{w}) = \mathbf{X}^\top \mathbf{W} \mathbf{X} = \mathbf{A} \in R^{m \times m}.
            \end{equation}
        $$

        Notation $\mathbf{W} \in R^{b \times b}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l_{\xi}}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} \in R^{b \times b},
            \end{equation}
        $$
        where $\mathbf{A} \in R^{p \times q}$ and $\mathbf{B} \in R^{s \times t}$ (where $s =\frac{b}{p}$ and $t = \frac{b}{q}$) are partitioned and reshaped from the parameter vector $\mathbf{w}$.

        The required length of parameter vector of this interdependence function is $l_{\xi} = pq + \frac{b^2}{pq}$.

        Attributes
        ----------
        p : int
            Number of partitions in the input dimension.
        q : int
            Number of partitions in the output dimension.

        Methods
        -------
        __init__(...)
            Initializes the hierarchical mapping parameterized bilinear interdependence function.
    """
    def __init__(self, p: int, q: int = None, name: str = 'hm_parameterized_bilinear_interdependence', *args, **kwargs):
        """
            Initializes the hierarchical mapping parameterized bilinear interdependence function.

            Parameters
            ----------
            p : int
                Number of partitions in the input dimension.
            q : int, optional
                Number of partitions in the output dimension. Defaults to `p`.
            name : str, optional
                Name of the interdependence function. Defaults to 'hm_parameterized_bilinear_interdependence'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """

        super().__init__(name=name, *args, **kwargs)

        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = hm_reconciliation(p=self.p, q=self.q)


class lphm_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    r"""
        A low-rank hierarchical mapping (LPHM) parameterized bilinear interdependence function.

        This class models interdependence using low-rank approximations with hierarchical mapping,
        where the parameter fabrication aligns with the LPHM methodology.

        Notes
        ----------
        Formally, given a data batch $\mathbf{X} \in R^{b \times m}$, we can represent the parameterized bilinear form-based interdependence function as follows:

        $$
            \begin{equation}\label{equ:bilinear_interdependence_function}
            \xi(\mathbf{X} | \mathbf{w}) = \mathbf{X}^\top \mathbf{W} \mathbf{X} = \mathbf{A} \in R^{m \times m}.
            \end{equation}
        $$

        Notation $\mathbf{W} \in R^{b \times b}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l_{\xi}}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = \mathbf{A} \otimes ( \mathbf{S} \mathbf{T}^\top) \in R^{b \times b},
            \end{equation}
        $$
        where $\mathbf{A} \in R^{p \times q}$, $\mathbf{S} \in R^{\frac{b}{p} \times r}$ and $\mathbf{T} \in R^{\frac{b}{q} \times r}$ are partitioned and reshaped from the parameter vector $\mathbf{w}$.

        The required length of parameter vector of this interdependence function is $l_{\xi} = pq + r \times (\frac{b}{p} + \frac{b}{q})$.

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
            Initializes the low-rank hierarchical mapping parameterized bilinear interdependence function.
    """
    def __init__(self, r: int, p: int, q: int = None, name: str = 'lphm_parameterized_bilinear_interdependence', *args, **kwargs):
        """
            Initializes the low-rank hierarchical mapping parameterized bilinear interdependence function.

            Parameters
            ----------
            r : int
                Rank of the low-rank approximation.
            p : int
                Number of partitions in the input dimension.
            q : int, optional
                Number of partitions in the output dimension. Defaults to `p`.
            name : str, optional
                Name of the interdependence function. Defaults to 'lphm_parameterized_bilinear_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            ValueError
                If the interdependence type is not supported.
            AssertionError
                If the dimensions are not divisible by the partitions.
        """

        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class dual_lphm_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    r"""
        A dual low-rank hierarchical mapping (Dual-LPHM) parameterized bilinear interdependence function.

        This class extends the LPHM methodology to model dual low-rank approximations
        for interdependence matrices.

        Notes
        ----------
        Formally, given a data batch $\mathbf{X} \in R^{b \times m}$, we can represent the parameterized bilinear form-based interdependence function as follows:

        $$
            \begin{equation}\label{equ:bilinear_interdependence_function}
            \xi(\mathbf{X} | \mathbf{w}) = \mathbf{X}^\top \mathbf{W} \mathbf{X} = \mathbf{A} \in R^{m \times m}.
            \end{equation}
        $$

        Notation $\mathbf{W} \in R^{b \times b}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l_{\xi}}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \mathbf{A} \otimes \mathbf{B} = ( \mathbf{P} \mathbf{Q}^\top) \otimes ( \mathbf{S} \mathbf{T}^\top) \in R^{b \times b},
            \end{equation}
        $$
        where $\mathbf{P} \in R^{p \times r}$, $\mathbf{Q} \in R^{q \times r}$,  $\mathbf{S} \in R^{\frac{b}{p} \times r}$ and $\mathbf{T} \in R^{\frac{b}{q} \times r}$ are partitioned and reshaped from the parameter vector $\mathbf{w}$.

        The required length of parameter vector of this interdependence function is $l_{\xi} = r \times (p + q + \frac{b}{p} + \frac{b}{q})$.

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
            Initializes the dual low-rank hierarchical mapping parameterized bilinear interdependence function.
    """
    def __init__(self, r: int, p: int, q: int = None, name: str = 'dual_lphm_parameterized_bilinear_interdependence', *args, **kwargs):
        """
            Initializes the dual low-rank hierarchical mapping parameterized bilinear interdependence function.

            Parameters
            ----------
            r : int
                Rank of the low-rank approximation.
            p : int
                Number of partitions in the input dimension.
            q : int, optional
                Number of partitions in the output dimension. Defaults to `p`.
            name : str, optional
                Name of the interdependence function. Defaults to 'dual_lphm_parameterized_bilinear_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            ValueError
                If the interdependence type is not supported.
            AssertionError
                If the dimensions are not divisible by the partitions.
        """

        super().__init__(name=name, *args, **kwargs)

        self.r = r
        self.p = p
        self.q = q if q is not None else p

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.m, self.calculate_m_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.b, self.calculate_b_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')
        assert d % self.p == 0 and d_prime % self.q == 0

        self.parameter_fabrication = dual_lphm_reconciliation(p=self.p, q=self.q, r=self.r)


class random_matrix_adaption_parameterized_bilinear_interdependence(parameterized_bilinear_interdependence):
    r"""
        A random matrix adaptation parameterized bilinear interdependence function.

        This class uses random matrix adaptation to compute bilinear interdependence matrices,
        where the random matrix serves as a low-rank approximation.

        Notes
        ----------
        Formally, given a data batch $\mathbf{X} \in R^{b \times m}$, we can represent the parameterized bilinear form-based interdependence function as follows:

        $$
            \begin{equation}\label{equ:bilinear_interdependence_function}
            \xi(\mathbf{X} | \mathbf{w}) = \mathbf{X}^\top \mathbf{W} \mathbf{X} = \mathbf{A} \in R^{m \times m}.
            \end{equation}
        $$

        Notation $\mathbf{W} \in R^{b \times b}$ denotes the parameter matrix fabricated from the learnable parameter vector $\mathbf{w} \in R^{l_{\xi}}$,
        which can be represented as follows:

        $$
            \begin{equation}
            \psi(\mathbf{w}) = \Lambda_1 \mathbf{A} \Lambda_1 \mathbf{B}^\top \in R^{b \times b},
            \end{equation}
        $$

        Notations $\Lambda_1$ and $\Lambda_2$ denote two diagonal matrices $\Lambda_1 = diag( \lambda_1) \in R^{m \times m}$ and $\Lambda_2 = diag(\lambda_2) \in R^{r \times r}$
        where the diagonal vectors $\lambda_1$ and $\lambda_2$ are partitioned from the parameter vector $\mathbf{w}$.
        Matrices $\mathbf{A} \in R^{b \times r}$ and $\mathbf{B} \in R^{r \times b}$ are randomly sampled from the Gaussian distribution $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

        The required length of parameter vector of this interdependence function is $\mathbf{w}$ is $l_{\xi} = b + r$.

        Attributes
        ----------
        r : int
            Rank of the random matrix approximation.

        Methods
        -------
        __init__(...)
            Initializes the random matrix adaptation parameterized bilinear interdependence function.
    """
    def __init__(self, r: int = 2, name: str = 'random_matrix_adaption_parameterized_bilinear_interdependence', *args, **kwargs):
        """
            Initializes the random matrix adaptation parameterized bilinear interdependence function.

            Parameters
            ----------
            r : int, optional
                Rank of the random matrix approximation. Defaults to 2.
            name : str, optional
                Name of the interdependence function. Defaults to 'random_matrix_adaption_parameterized_bilinear_interdependence'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(name=name, *args, **kwargs)
        self.r = r
        self.parameter_fabrication = random_matrix_adaption_reconciliation(r=self.r)
