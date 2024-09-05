# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
import warnings

###############################
# Topological Interdependency #
###############################

from tinybig.interdependence import interdependence
from tinybig.koala.topology import chain as chain_structure
from tinybig.koala.topology import graph as graph_structure
from tinybig.koala.topology import grid as grid_structure
from tinybig.koala.linear_algebra import accumulative_matrix_power, matrix_power, sparse_matrix_to_torch_sparse_tensor

from numpy.linalg import inv
from scipy.linalg import expm
import scipy.sparse as sp

class grid_interdependence(interdependence):
    pass



class chain_interdependence(interdependence):

    def __init__(self, name: str = 'chain_interdependence', topological_structure: chain_structure = None,
                 links: dict | list = None, length: int = None, bi_directional: bool = False, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        if topological_structure is not None:
            self.topological_structure = topological_structure
        elif links is not None or length is not None:
            self.topological_structure = chain_structure(links=links, length=length, bi_directional=bi_directional)
        else:
            raise ValueError('Either nodes or links must be provided')

        self.o = self.topological_structure.order()
        self.o_prime = self.o

        self.mappings = None

    def forward(self, normalization: bool = False, mode: str = 'row-column', self_dependence: bool = False, device: str = 'cpu', *args, **kwargs):
        adj = self.topological_structure.to_matrix(normalization=normalization, mode=mode)

        if self_dependence:
            adj += sp.eye(adj.shape[0])
        A = sparse_matrix_to_torch_sparse_tensor(adj)

        assert A.shape == (self.o, self.o_prime)
        return self.post_process(x=A, device=device)


class multihop_chain_interdependence(chain_interdependence):

    def __init__(self, name: str = 'multihop_chain_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def forward(self, h: int = 1, accumulative: bool = False, normalization: bool = False, mode: str = 'row-column', self_dependence: bool = False, device: str = 'cpu', *args, **kwargs):
        adj = self.topological_structure.to_matrix(normalization=normalization, mode=mode)

        if accumulative:
            A = accumulative_matrix_power(adj, h)
        else:
            A = matrix_power(adj, h)

        if self_dependence:
            A += sp.eye(adj.shape[0])

        A = sparse_matrix_to_torch_sparse_tensor(A)

        assert A.shape == (self.o, self.o_prime)
        return self.post_process(x=A, device=device)


class approx_multihop_chain_interdependence(interdependence):

    def __init__(self, name: str = 'approx_multihop_chain_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def forward(self, normalization: bool = False, mode: str = 'row-column', approx_type: str = 'reciprocal', device: str = 'cpu', *args, **kwargs):
        adj = self.topological_structure.to_matrix(normalization=normalization, mode=mode)

        if approx_type == 'reciprocal':
            A = inv(sp.eye(adj.shape[0]) - adj)
        elif approx_type == 'exponential':
            A = expm(adj)
        else:
            raise ValueError('Approx type must be either "reciprocal" or "exponential"')

        A = sparse_matrix_to_torch_sparse_tensor(A)

        assert A.shape == (self.o, self.o_prime)
        return self.post_process(x=A, device=device)



class graph_interdependence(interdependence):

    def __init__(self, name: str = 'graph_interdependence', topological_structure: graph_structure = None, nodes: dict | list = None, links: dict | list = None, directed: bool = True, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        if topological_structure is not None:
            self.topological_structure = topological_structure
        elif nodes is not None and links is not None:
            self.topological_structure = graph_structure(nodes=nodes, links=links, directed=directed)
        else:
            raise ValueError('Either nodes or links must be provided')

        self.o = self.topological_structure.order()
        self.o_prime = self.o

        self.mappings = None

    def get_node_index_mappings(self):
        if self.mappings is None:
            warnings.warn("The mapping has not been assigned yet, please call the forward method first...")
        return self.mappings

    def forward(self, normalization: bool = False, mode: str = 'row-column', self_dependence: bool = True, device: str = 'cpu', *args, **kwargs):
        adj, self.mappings = self.topological_structure.to_matrix(normalization=normalization, mode=mode)

        if self_dependence:
            adj += sp.eye(adj.shape[0])
        A = sparse_matrix_to_torch_sparse_tensor(adj)

        assert A.shape == (self.o, self.o_prime)
        return self.post_process(x=A, device=device)


class multihop_graph_interdependence(graph_interdependence):

    def __init__(self, name: str = 'multihop_graph_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def forward(self, h: int = 1, accumulative: bool = False, normalization: bool = False, mode: str = 'row-column', self_dependence: bool = True, device: str = 'cpu', *args, **kwargs):
        adj, self.mappings = self.topological_structure.to_matrix(normalization=normalization, mode=mode)

        if accumulative:
            A = accumulative_matrix_power(adj, h)
        else:
            A = matrix_power(adj, h)

        if self_dependence:
            A += sp.eye(adj.shape[0])
        A = sparse_matrix_to_torch_sparse_tensor(A)

        assert A.shape == (self.o, self.o_prime)
        return self.post_process(x=A, device=device)


class pagerank_multihop_graph_interdependence(graph_interdependence):

    def __init__(self, name: str = 'pagerank_multihop_graph_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def forward(self, c: float = 0.15, normalization: bool = False, mode: str = 'row-column', device: str = 'cpu', *args, **kwargs):
        adj, self.mappings = self.topological_structure.to_matrix(normalization=normalization, mode=mode)

        A = c * inv((sp.eye(adj.shape[0]) - (1.0 - c) * adj).toarray())
        A = sparse_matrix_to_torch_sparse_tensor(A)

        assert A.shape == (self.o, self.o_prime)
        return self.post_process(x=A, device=device)
