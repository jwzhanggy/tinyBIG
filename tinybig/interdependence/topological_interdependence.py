# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Topological Interdependence #
###############################

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

    def __init__(
        self,
        b: int, m: int,
        interdependence_type: str = 'instance',
        name: str = 'chain_interdependence',
        chain: chain_structure = None,
        length: int = None, bi_directional: bool = False,
        normalization: bool = False, normalization_mode: str = 'row_column', self_dependence: bool = True,
        require_data: bool = False, require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data, require_parameters=require_parameters, device=device, *args, **kwargs)

        if chain is not None:
            self.chain = chain
        elif length is not None:
            self.chain = chain_structure(length=length, bi_directional=bi_directional)
        else:
            raise ValueError('Either chain structure of chain length must be provided...')

        self.node_id_index_map = None
        self.node_index_id_map = None

        self.normalization = normalization
        self.normalization_mode = normalization_mode
        self.self_dependence = self_dependence

    def is_bi_directional(self):
        return not self.chain.is_directed()

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.chain.to_matrix(normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)
            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            if self.self_dependence:
                adj += torch.eye(adj.shape[0], device=device)

            A = self.post_process(x=adj, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A

            return A


class multihop_chain_interdependence(chain_interdependence):

    def __init__(self, h: int = 1, accumulative: bool = False, name: str = 'multihop_chain_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.h = h
        self.accumulative = accumulative

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.chain.to_matrix(normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)
            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            if self.accumulative:
                A = accumulative_matrix_power(adj, self.h)
                if self.is_bi_directional():
                    A = degree_based_normalize_matrix(A, mode='column')
            else:
                A = matrix_power(adj, self.h)

            if self.self_dependence:
                A += torch.eye(A.shape[0], device=device)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A

            return A


class inverse_approx_multihop_chain_interdependence(chain_interdependence):

    def __init__(self, name: str = 'inverse_approx_multihop_chain_interdependence', normalization: bool = False, normalization_mode: str = 'row_column', *args, **kwargs):
        super().__init__(name=name, normalization=normalization, normalization_mode=normalization_mode, *args, **kwargs)

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
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

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A

            return A


class exponential_approx_multihop_chain_interdependence(chain_interdependence):

    def __init__(self, name: str = 'exponential_approx_multihop_chain_interdependence', normalization: bool = False, normalization_mode: str = 'row_column', *args, **kwargs):
        super().__init__(name=name, normalization=normalization, normalization_mode=normalization_mode, *args, **kwargs)

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.chain.to_matrix(normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)
            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            if adj.device.type == 'mps':
                A = torch.matrix_exp(adj.to('cpu')).to('mps')
            else:
                A = torch.matrix_exp(adj)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A

            return A


class graph_interdependence(interdependence):

    def __init__(
        self,
        b: int, m: int,
        interdependence_type: str = 'instance',
        name: str = 'graph_interdependence',
        graph: graph_structure = None,
        nodes: list = None, links: list = None, directed: bool = True,
        normalization: bool = False, normalization_mode: str = 'row_column',
        self_dependence: bool = False,
        require_data: bool = False, require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
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
        if self.node_index_id_map is None:
            warnings.warn("The mapping has not been assigned yet, please call the calculate_A method first...")
        return self.node_index_id_map

    def get_node_id_index_map(self):
        if self.node_id_index_map is None:
            warnings.warn("The mapping has not been assigned yet, please call the calculate_A method first...")
        return self.node_id_index_map

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.graph.to_matrix(normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)

            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            if self.self_dependence:
                adj += torch.eye(adj.shape[0], device=device)

            A = self.post_process(x=adj, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A
            return A


class multihop_graph_interdependence(graph_interdependence):

    def __init__(self, h: int = 1, accumulative: bool = False, name: str = 'multihop_graph_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.h = h
        self.accumulative = accumulative

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            adj, mappings = self.graph.to_matrix(normalization=self.normalization, normalization_mode=self.normalization_mode, device=device)

            self.node_id_index_map = mappings['node_id_index_map']
            self.node_index_id_map = mappings['node_index_id_map']

            if self.accumulative:
                A = accumulative_matrix_power(adj, self.h)
            else:
                A = matrix_power(adj, self.h)

            if self.self_dependence:
                A += torch.eye(adj.shape[0], device=device)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A
            return A


class pagerank_multihop_graph_interdependence(graph_interdependence):

    def __init__(self, c: float = 0.15, name: str = 'pagerank_multihop_graph_interdependence', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.c = c

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
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

            if not self.require_data and not self.require_parameters and self.A is not None:
                self.A = A
            return A
