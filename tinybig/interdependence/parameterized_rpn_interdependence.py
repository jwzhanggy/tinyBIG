# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################
# Parameterized RPN Interdependence #
#####################################
"""
The parameterized rpn based interdependence functions

This module contains the parameterized rpn based interdependence function.
"""

import torch

from tinybig.interdependence import interdependence
import tinybig.module.base_transformation as base_transformation
import tinybig.module.base_fabrication as base_fabrication


class parameterized_rpn_interdependence(interdependence):
    r"""
        A parameterized Random Projection Network (RPN) interdependence function.

        This class computes interdependence matrices using a combination of data transformations
        and parameter fabrications, designed for flexible random projection-based modeling.

        Notes
        ----------
        Formally, the interdependence function $\xi: R^{b \times m} \to R^{m \times m'}$ can be viewed as a mapping between the input vector space of dimension $(b \times m)$ and the output vector space of dimension $(m \times m')$. We propose to represent it as follows:

        $$
            \begin{equation}
            \xi(\mathbf{X} | \mathbf{w}) = \left\langle \kappa'(\mathbf{x}) , \psi'(\mathbf{w}') \right\rangle,
            \end{equation}
        $$

        where $\mathbf{x} = \text{flatten}(\mathbf{X})$ is the flattened vector of length $(b \times m)$ from the input data batch matrix $\mathbf{X} \in R^{b \times m}$. This is viewed as a single (independent) pseudo ``data instance'' for the above {\old}-layer. The notations are defined as:

        $$
            \begin{equation}
            \kappa': R^{(b \times m)} \to R^D \text{, and } \psi': R^{l} \to R^{(m \times m') \times D},
            \end{equation}
        $$
        
        which denote the data expansion function and parameter reconciliation function used for defining the interdependence function, respectively.


        Attributes
        ----------
        data_transformation : base_transformation
            The data transformation module used to preprocess the input data.
        parameter_fabrication : base_fabrication
            The parameter fabrication module used to compute the projection matrix.
        b_prime : int
            Number of rows in the output interdependence matrix.
        m_prime : int
            Number of columns in the output interdependence matrix.

        Methods
        -------
        calculate_l()
            Computes the total number of parameters needed.
        calculate_b_prime(b=None)
            Computes the effective number of rows in the interdependence matrix.
        calculate_m_prime(m=None)
            Computes the effective number of columns in the interdependence matrix.
        calculate_A(x=None, w=None, device='cpu', *args, **kwargs)
            Computes the parameterized RPN interdependence matrix.
    """

    def __init__(
        self,
        b: int, m: int,
        data_transformation: base_transformation,
        parameter_fabrication: base_fabrication,
        b_prime: int = None, m_prime: int = None,
        interdependence_type: str = 'attribute',
        name: str = 'parameterized_rpn_interdependence',
        require_parameters: bool = True,
        require_data: bool = True,
        device: str = 'cpu', *args, **kwargs
    ):
        """
            Initializes the parameterized RPN interdependence function.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            data_transformation : base_transformation
                A module for transforming input data before projection.
            parameter_fabrication : base_fabrication
                A module for fabricating parameters for the projection matrix.
            b_prime : int, optional
                Number of rows in the output interdependence matrix. Defaults to `b`.
            m_prime : int, optional
                Number of columns in the output interdependence matrix. Defaults to `m`.
            interdependence_type : str, optional
                Type of interdependence ('instance', 'attribute', etc.). Defaults to 'attribute'.
            name : str, optional
                Name of the interdependence function. Defaults to 'parameterized_rpn_interdependence'.
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

            Raises
            ------
            ValueError
                If `data_transformation` or `parameter_fabrication` is not specified.
        """
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data,
                         require_parameters=require_parameters, device=device, *args, **kwargs)

        if data_transformation is None or parameter_fabrication is None:
            raise ValueError('data_transformation or parameter_fabrication must be specified...')

        self.data_transformation = data_transformation
        self.parameter_fabrication = parameter_fabrication
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
            AssertionError
                If the input and output dimensions are not specified.
        """
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            d, d_prime = self.m, self.calculate_b_prime()
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            d, d_prime = self.b, self.calculate_m_prime()
        else:
            raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

        assert d is not None and d_prime is not None
        D = self.data_transformation.calculate_D(m=d)
        return self.parameter_fabrication.calculate_l(n=d_prime, D=D)

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
            Computes the parameterized RPN interdependence matrix.

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

            self.data_transformation.to(device)
            self.parameter_fabrication.to(device)

            if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                d, d_prime = self.m, self.calculate_b_prime()
            elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                d, d_prime = self.b, self.calculate_m_prime()
            else:
                raise ValueError(f'Interdependence type {self.interdependence_type} not supported')

            kappa_x = self.data_transformation(x.t(), device=device)
            D = self.data_transformation.calculate_D(m=d)
            phi_w = self.parameter_fabrication(w=w, n=d_prime, D=D, device=device)

            A = torch.matmul(kappa_x, phi_w.T)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A
