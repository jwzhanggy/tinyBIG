# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################################
# Test Topology Structures in koala.topology #
##############################################

import pytest
import torch
from tinybig.koala.topology import base_topology, graph, chain


class TestBaseTopology:

    @pytest.fixture
    def sample_topology(self):
        nodes = ['A', 'B', 'C', 'D']
        links = [('A', 'B'), ('B', 'C'), ('C', 'D')]
        return base_topology(nodes=nodes, links=links, directed=False)

    def test_initialization(self, sample_topology):
        assert sample_topology.get_name() == 'base_topological_structure'
        assert sample_topology.get_node_num() == 4
        assert sample_topology.get_link_num() == 3

    def test_add_node(self, sample_topology):
        sample_topology.add_node('E')
        assert 'E' in sample_topology.get_nodes()
        assert sample_topology.get_node_num() == 5

    def test_add_link(self, sample_topology):
        sample_topology.add_link(('D', 'A'))
        assert ('D', 'A') in sample_topology.get_links()
        assert sample_topology.get_link_num() == 4

    def test_delete_node(self, sample_topology):
        sample_topology.delete_node('A')
        assert 'A' not in sample_topology.get_nodes()
        assert sample_topology.get_node_num() == 3

    def test_delete_link(self, sample_topology):
        sample_topology.delete_link(('A', 'B'))
        assert ('A', 'B') not in sample_topology.get_links()
        assert sample_topology.get_link_num() == 2

    def test_get_node_attribute(self, sample_topology):
        sample_topology.node_attributes = {'A': 'attr1'}
        assert sample_topology.get_node_attribute('A') == 'attr1'
        assert sample_topology.get_node_attribute('B') is None

    def test_to_matrix(self, sample_topology):
        matrix, metadata = sample_topology.to_matrix()
        assert isinstance(matrix, torch.Tensor)
        assert matrix.shape == (4, 4)

    def test_normalized_matrix(self, sample_topology):
        matrix, metadata = sample_topology.to_matrix(normalization=True, normalization_mode='column')
        assert torch.isclose(matrix.sum(dim=0), torch.tensor(1.0)).all()


class TestChain:

    def test_chain_initialization(self):
        chain_topology = chain(length=4)
        assert chain_topology.get_node_num() == 4
        assert chain_topology.get_link_num() == 4
        assert chain_topology.length() == 4

    def test_chain_links(self):
        chain_topology = chain(length=4)
        expected_links = [(0, 1), (1, 2), (2, 3), (3, 4)]
        assert chain_topology.get_links() == expected_links


class TestGraph:

    @pytest.fixture
    def graph_topology(self):
        nodes = ['A', 'B', 'C', 'D']
        links = [('A', 'B'), ('B', 'C'), ('C', 'D')]
        return graph(nodes=nodes, links=links)

    def test_graph_initialization(self, graph_topology):
        assert graph_topology.get_node_num() == 4
        assert graph_topology.get_link_num() == 3

    def test_bfs(self, graph_topology):
        # Test BFS (you'll need to implement the method)
        pass

    def test_dfs(self, graph_topology):
        # Test DFS (you'll need to implement the method)
        pass
