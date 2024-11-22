# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Topological Interdependence #
###############################
"""
The topological interdependence functions

This module contains the topological interdependence functions, including
    graph_interdependence,
    multihop_graph_interdependence,
    pagerank_multihop_graph_interdependence,
    chain_interdependence,
    multihop_chain_interdependence,
    inverse_approx_multihop_chain_interdependence,
    exponential_approx_multihop_chain_interdependence
"""

import warnings
import torch

from tinybig.interdependence import interdependence
from tinybig.koala.topology import chain as chain_structure
from tinybig.koala.topology import graph as graph_structure
from tinybig.koala.linear_algebra import (
    accumulative_matrix_power,
    matrix_power,
    degree_based_normalize_matrix
)


class chain_interdependence(interdependence):
    r"""
        A chain-based interdependence function.

        This class computes the interdependence matrix using a chain structure.

        Notes
        ----------
        Formally, given a data instance $\mathbf{x} \in R^{m}$ with sequential chain-structured interdependence relationships among the attributes of length $m$,
        we can define the corresponding unidirectional chain interdependence function as follows:

        $$
            \begin{equation}
            \xi(\mathbf{x}) = \mathbf{A}  \in R^{m \times m'},
            \end{equation}
        $$

        where $\mathbf{A}$ is the composed attribute interdependence matrix. By default, the output dimension $m'$ equals the input instance dimension, {\ie} $m' = m$.

        In many cases, we sum this interdependence matrix with an identity matrix to denote self-dependency:

        $$
            \begin{equation}
            \xi(\mathbf{x}) = \mathbf{A} + \mathbf{I}  \in R^{m \times m}.
            \end{equation}
        $$

        Here, $\mathbf{I} \in R^{m \times m}$ is a square diagonal identity matrix of size $m \times m$, allowing the function to model both interdependence and self-dependence with a single dependency function. This self-dependence can also be defined using the linear remainder term in {\our}, both of which contribute to defining sequential interdependence relationships.


        Attributes
        ----------
        chain : chain_structure
            The chain structure representing the interdependence.
        normalization : bool
            Whether to normalize the interdependence matrix.
        normalization_mode : str
            The mode of normalization ('row', 'column', etc.).
        self_dependence : bool
            Whether nodes are self-dependent.
        self_scaling : float
            Scaling factor for self-dependence.

        Methods
        -------
        __init__(...)
            Initializes the chain-based interdependence function.
        is_bi_directional()
            Checks if the chain is bidirectional.
        calculate_A(...)
            Computes the interdependence matrix.
    """
    def __init__(
        self,
        b: int, m: int,
        interdependence_type: str = 'instance',
        name: str = 'chain_interdependence',
        chain: chain_structure = None,
        chain_length: int = None, bi_directional: bool = False,
        normalization: bool = False, normalization_mode: str = 'row',
        self_dependence: bool = True, self_scaling: float = 1.0,
        require_data: bool = False, require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        """
            Initializes the chain-based interdependence function.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            interdependence_type : str, optional
                Type of interdependence ('instance', 'attribute', etc.). Defaults to 'instance'.
            name : str, optional
                Name of the interdependence function. Defaults to 'chain_interdependence'.
            chain : chain_structure, optional
                Predefined chain structure. Defaults to None.
            chain_length : int, optional
                Length of the chain structure. Required if `chain` is not provided. Defaults to None.
            bi_directional : bool, optional
                Whether the chain is bidirectional. Defaults to False.
            normalization : bool, optional
                Whether to normalize the interdependence matrix. Defaults to False.
            normalization_mode : str, optional
                The mode of normalization ('row', 'column', etc.). Defaults to 'row'.
            self_dependence : bool, optional
                Whether nodes are self-dependent. Defaults to True.
            self_scaling : float, optional
                Scaling factor for self-dependence. Defaults to 1.0.
            require_data : bool, optional
                Whether the interdependence function requires data. Defaults to False.
            require_parameters : bool, optional
                Whether the interdependence function requires parameters. Defaults to False.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            ValueError
                If neither `chain` nor `chain_length` is provided.
        """
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data, require_parameters=require_parameters, device=device, *args, **kwargs)

        if chain is not None:
            self.chain = chain
        elif chain_length is not None:
            self.chain = chain_structure(length=chain_length, bi_directional=bi_directional)
        else:
            raise ValueError('Either chain structure of chain length must be provided...')

        self.node_id_index_map = None
        self.node_index_id_map = None

        self.normalization = normalization
        self.normalization_mode = normalization_mode
        self.self_dependence = self_dependence
        self.self_scaling = self_scaling

    def is_bi_directional(self):
        """
            Checks if the chain is bidirectional.

            Returns
            -------
            bool
                True if the chain is bidirectional, False otherwise.
        """
        return not self.chain.is_directed()

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the interdependence matrix using the chain structure.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor. Defaults to None.
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
            AssertionError
                If the computed matrix shape is invalid.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.chain.to_matrix(
                self_dependence=self.self_dependence,
                self_scaling=self.self_scaling,
                normalization=self.normalization,
                normalization_mode=self.normalization_mode,
                device=device
            )

            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            A = self.post_process(x=adj, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A


class multihop_chain_interdependence(chain_interdependence):
    r"""
        A multihop chain-based interdependence function.

        Notes
        ----------
        To accumulate all data instances within $h$-hops, we introduce the accumulative multi-hop chain-based structural interdependence function as follows:

        $$
            \begin{equation}\label{equ:chain_accumulative}
            \xi(\mathbf{x} | 0:h) = \mathbf{I} + \mathbf{A} + \mathbf{A}^2 + \mathbf{A}^3 + \cdots + \mathbf{A}^h = \sum_{i=0}^h \mathbf{A}^i \in R^{m \times m}.
            \end{equation}
        $$

        Attributes
        ----------
        h : int
            Number of hops to consider in the chain.
        accumulative : bool
            Whether to accumulate interdependence over multiple hops.

        Methods
        -------
        __init__(...)
            Initializes the multihop chain interdependence function.
        calculate_A(...)
            Computes the multihop interdependence matrix.
    """
    def __init__(self, h: int = 1, accumulative: bool = False, name: str = 'multihop_chain_interdependence', *args, **kwargs):
        """
            Initializes the multihop chain interdependence function.

            Parameters
            ----------
            h : int, optional
                Number of hops to consider. Defaults to 1.
            accumulative : bool, optional
                Whether to accumulate interdependence over multiple hops. Defaults to False.
            name : str, optional
                Name of the interdependence function. Defaults to 'multihop_chain_interdependence'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.h = h
        self.accumulative = accumulative

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the multihop interdependence matrix.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor. Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The computed multihop interdependence matrix.

            Raises
            ------
            AssertionError
                If the computed matrix shape is invalid.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.chain.to_matrix(self_dependence=self.self_dependence, normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)
            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            if self.accumulative:
                A = accumulative_matrix_power(adj, self.h)
                if self.is_bi_directional():
                    A = degree_based_normalize_matrix(A, mode='column')
            else:
                A = matrix_power(adj, self.h)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A

            return A


class inverse_approx_multihop_chain_interdependence(chain_interdependence):
    r"""
        An approximate multihop chain interdependence function using matrix inversion.

        Notes
        -------
        To further optimize the computations, we propose approximating the accumulative interdependence function using Taylor's polynomial expansion series.
        Considering the Taylor's expansions of the reciprocal function $\frac{1}{1-x}$:

        $$
            \begin{equation}
            \frac{1}{1-x} = 1 + x + x^2 + x^3 + \cdots = \sum_{h=0}^{\infty} x^h.
            \end{equation}
        $$

        Based on them, we define the reciprocal structural interdependence function and exponential structural interdependence function for approximating the above multi-hop chain-structured topological interdependence relationships as:

        $$
            \begin{equation}
            \xi(\mathbf{x}) = (\mathbf{I} - \mathbf{A})^{-1} \in R^{m \times m} \text{, and } \xi(\mathbf{x}) = \exp(\mathbf{A}) \in R^{m \times m}.
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the approximate multihop chain interdependence function.
        calculate_A(...)
            Computes the approximate multihop interdependence matrix.
    """
    def __init__(self, name: str = 'inverse_approx_multihop_chain_interdependence', normalization: bool = False, normalization_mode: str = 'row', *args, **kwargs):
        """
            Initializes the approximate multihop chain interdependence function using matrix inversion.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'inverse_approx_multihop_chain_interdependence'.
            normalization : bool, optional
                Whether to normalize the interdependence matrix. Defaults to False.
            normalization_mode : str, optional
                The mode of normalization ('row', 'column', etc.). Defaults to 'row'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, normalization=normalization, normalization_mode=normalization_mode, *args, **kwargs)

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the approximate multihop interdependence matrix using matrix inversion.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor. Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The computed approximate multihop interdependence matrix.

            Raises
            ------
            AssertionError
                If the computed matrix shape is invalid.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.chain.to_matrix(normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)
            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            if self.is_bi_directional():
                A = accumulative_matrix_power(adj, n=adj.shape[0] - 1)
                A = degree_based_normalize_matrix(mx=A, mode='column')
            else:
                A = torch.inverse(torch.eye(adj.shape[0], device=device) - adj)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A

            return A


class exponential_approx_multihop_chain_interdependence(chain_interdependence):
    r"""
        An approximate multihop chain interdependence function using matrix exponentiation.

        This class computes the interdependence matrix by approximating multihop relationships
        through matrix exponentiation.

        Notes
        -------
        To further optimize the computations, we propose approximating the accumulative interdependence function using Taylor's polynomial expansion series.
        Considering the Taylor's expansions of the exponential function $\exp(x)$:

        $$
            \begin{equation}
            \exp(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots = \sum_{h=0}^{\infty} \frac{x^h}{h!}.
            \end{equation}
        $$

        Based on them, we define the reciprocal structural interdependence function and exponential structural interdependence function for approximating the above multi-hop chain-structured topological interdependence relationships as:

        $$
            \begin{equation}
            \xi(\mathbf{x}) = \exp(\mathbf{A}) \in R^{m \times m}.
            \end{equation}
        $$

        Methods
        -------
        __init__(...)
            Initializes the approximate multihop chain interdependence function.
        calculate_A(...)
            Computes the approximate multihop interdependence matrix using matrix exponentiation.
    """
    def __init__(self, name: str = 'exponential_approx_multihop_chain_interdependence', normalization: bool = False, normalization_mode: str = 'row', *args, **kwargs):
        """
            Initializes the approximate multihop chain interdependence function using matrix exponentiation.

            Parameters
            ----------
            name : str, optional
                Name of the interdependence function. Defaults to 'exponential_approx_multihop_chain_interdependence'.
            normalization : bool, optional
                Whether to normalize the interdependence matrix. Defaults to False.
            normalization_mode : str, optional
                The mode of normalization ('row', 'column', etc.). Defaults to 'row'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, normalization=normalization, normalization_mode=normalization_mode, *args, **kwargs)

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the approximate multihop interdependence matrix using matrix exponentiation.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor. Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The computed approximate multihop interdependence matrix.

            Raises
            ------
            AssertionError
                If the computed matrix shape is invalid.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.chain.to_matrix(normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)
            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            if adj.device.type == 'mps':
                A = torch.matrix_exp(adj.to('cpu')).to('mps')
            else:
                A = torch.matrix_exp(adj.to_dense()).to_sparse_coo()

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A

            return A


class graph_interdependence(interdependence):
    r"""
        A graph-based interdependence function.

        This class computes the interdependence matrix using a graph structure, allowing
        relationships to be modeled based on nodes and edges in the graph.

        Notes
        ----------
        Based on the graph structure, we define the graph interdependence function as:

        $$
            \begin{equation}\label{equ:graph_interdependence_function}
            \xi(\mathbf{x} | G) = \mathbf{A} \in R^{m \times m'},
            \end{equation}
        $$

        where the output dimension $m' = m$ by default.

        Attributes
        ----------
        graph : graph_structure
            The graph structure representing the interdependence.
        node_id_index_map : dict
            Mapping of node IDs to their indices in the matrix.
        node_index_id_map : dict
            Mapping of matrix indices back to their corresponding node IDs.
        normalization : bool
            Whether to normalize the interdependence matrix.
        normalization_mode : str
            The mode of normalization ('row', 'column', etc.).
        self_dependence : bool
            Whether nodes are self-dependent.

        Methods
        -------
        __init__(...)
            Initializes the graph-based interdependence function.
        get_node_index_id_map()
            Retrieves the mapping from indices to node IDs.
        get_node_id_index_map()
            Retrieves the mapping from node IDs to indices.
        calculate_A(...)
            Computes the interdependence matrix using the graph structure.
    """

    def __init__(
        self,
        b: int, m: int,
        interdependence_type: str = 'instance',
        name: str = 'graph_interdependence',
        graph: graph_structure = None,
        nodes: list = None, links: list = None, directed: bool = True,
        normalization: bool = False, normalization_mode: str = 'row',
        self_dependence: bool = False,
        require_data: bool = False, require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        """
            Initializes the graph-based interdependence function.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            interdependence_type : str, optional
                Type of interdependence ('instance', 'attribute', etc.). Defaults to 'instance'.
            name : str, optional
                Name of the interdependence function. Defaults to 'graph_interdependence'.
            graph : graph_structure, optional
                Predefined graph structure. Defaults to None.
            nodes : list, optional
                List of nodes in the graph. Required if `graph` is not provided. Defaults to None.
            links : list, optional
                List of links (edges) in the graph. Required if `graph` is not provided. Defaults to None.
            directed : bool, optional
                Whether the graph is directed. Defaults to True.
            normalization : bool, optional
                Whether to normalize the interdependence matrix. Defaults to False.
            normalization_mode : str, optional
                The mode of normalization ('row', 'column', etc.). Defaults to 'row'.
            self_dependence : bool, optional
                Whether nodes are self-dependent. Defaults to False.
            require_data : bool, optional
                Whether the interdependence function requires data. Defaults to False.
            require_parameters : bool, optional
                Whether the interdependence function requires parameters. Defaults to False.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            ValueError
                If neither `graph` nor both `nodes` and `links` are provided.
        """
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data, require_parameters=require_parameters, device=device, *args, **kwargs)

        if graph is not None:
            self.graph = graph
        elif nodes is not None and links is not None:
            self.graph = graph_structure(nodes=nodes, links=links, directed=directed)
        else:
            raise ValueError('Either nodes or links must be provided')

        self.node_id_index_map = None
        self.node_index_id_map = None

        self.normalization = normalization
        self.normalization_mode = normalization_mode
        self.self_dependence = self_dependence

    def get_node_index_id_map(self):
        """
            Retrieves the mapping from indices to node IDs.

            Returns
            -------
            dict
                A dictionary mapping matrix indices to node IDs.

            Warnings
            --------
            If the mapping has not been assigned, a warning will be raised.
        """
        if self.node_index_id_map is None:
            warnings.warn("The mapping has not been assigned yet, please call the calculate_A method first...")
        return self.node_index_id_map

    def get_node_id_index_map(self):
        """
            Retrieves the mapping from node IDs to indices.

            Returns
            -------
            dict
                A dictionary mapping node IDs to matrix indices.

            Warnings
            --------
            If the mapping has not been assigned, a warning will be raised.
        """
        if self.node_id_index_map is None:
            warnings.warn("The mapping has not been assigned yet, please call the calculate_A method first...")
        return self.node_id_index_map

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the interdependence matrix using the graph structure.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor. Defaults to None.
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
            AssertionError
                If the computed matrix shape is invalid.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.graph.to_matrix(self_dependence=self.self_dependence, normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)

            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            A = self.post_process(x=adj, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A


class multihop_graph_interdependence(graph_interdependence):
    r"""
        A multihop graph-based interdependence function.

        Notes
        ----------

        To model multi-hop dependency relationships among data instances, we introduce the multi-hop graph interdependence function and the accumulative multi-hop graph interdependence function as follows:

        $$
            \begin{equation}
            \xi(\mathbf{x} | h) = \mathbf{A}^h \in R^{m \times m} \text{, and } \xi(\mathbf{x} | 0: h) = \sum_{i=0}^h \mathbf{A}^i \in R^{m \times m}.
            \end{equation}
        $$

        Attributes
        ----------
        h : int
            Number of hops to consider in the graph.
        accumulative : bool
            Whether to accumulate interdependence over multiple hops.

        Methods
        -------
        __init__(...)
            Initializes the multihop graph interdependence function.
        calculate_A(...)
            Computes the multihop interdependence matrix.
    """

    def __init__(self, h: int = 1, accumulative: bool = False, name: str = 'multihop_graph_interdependence', *args, **kwargs):
        """
            Initializes the multihop graph interdependence function.

            Parameters
            ----------
            h : int, optional
                Number of hops to consider. Defaults to 1.
            accumulative : bool, optional
                Whether to accumulate interdependence over multiple hops. Defaults to False.
            name : str, optional
                Name of the interdependence function. Defaults to 'multihop_graph_interdependence'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.h = h
        self.accumulative = accumulative

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the multihop interdependence matrix.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor. Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The computed multihop interdependence matrix.

            Raises
            ------
            AssertionError
                If the computed matrix shape is invalid.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.graph.to_matrix(self_dependence=self.self_dependence, normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)

            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            if self.accumulative:
                A = accumulative_matrix_power(adj, self.h)
            else:
                A = matrix_power(adj, self.h)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A


class pagerank_multihop_graph_interdependence(graph_interdependence):
    r"""
        A multihop graph-based interdependence function using the PageRank algorithm.

        Notes
        ----------

        In addition to these formulas that calculate powers of matrix $\mathbf{A}$, existing graph studies also offer other approaches to calculate long-distance dependency relationships among data instances, such as the PageRank algorithm.
        Without delving into the step-wise derivation of PageRank updating equations, we define the PageRank multi-hop graph interdependence function as follows:

        $$
            \begin{equation}
            \xi(\mathbf{x}) = \alpha \cdot \left( \mathbf{I} - (1- \alpha) \cdot {\mathbf{A}} \right)^{-1} \in R^{m \times m}.
            \end{equation}
        $$

        Here, $\alpha \in [0, 1]$ is a hyper-parameter of the function, typically set to $0.15$ by default. Usually, matrix $\mathbf{A}$ is normalized before being used in this formula.

        Attributes
        ----------
        c : float
            Damping factor for the PageRank algorithm.

        Methods
        -------
        __init__(...)
            Initializes the PageRank multihop graph interdependence function.
        calculate_A(...)
            Computes the interdependence matrix using the PageRank algorithm.
    """
    def __init__(self, c: float = 0.15, name: str = 'pagerank_multihop_graph_interdependence', *args, **kwargs):
        """
            Initializes the PageRank multihop graph interdependence function.

            Parameters
            ----------
            c : float, optional
                Damping factor for the PageRank algorithm. Defaults to 0.15.
            name : str, optional
                Name of the interdependence function. Defaults to 'pagerank_multihop_graph_interdependence'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.c = c

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Computes the interdependence matrix using the PageRank algorithm.

            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor. Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The computed interdependence matrix using the PageRank algorithm.

            Raises
            ------
            AssertionError
                If the computed matrix shape is invalid.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.graph.to_matrix(normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)
            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            A = self.c * torch.inverse((torch.eye(adj.shape[0], device=device) - (1.0 - self.c) * adj))

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A
