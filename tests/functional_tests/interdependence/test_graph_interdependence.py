# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Test Topological Interdependence #
###############################

import pytest
import torch
import numpy as np
from tinybig.interdependence.topological_interdependence import (
    chain_interdependence,
    multihop_chain_interdependence,
    approx_multihop_chain_interdependence,
    graph_interdependence,
    multihop_graph_interdependence,
    pagerank_multihop_graph_interdependence
)
from tinybig.koala.topology import chain as chain_structure
from tinybig.koala.topology import graph as graph_structure

device = 'mps'

@pytest.fixture
def sample_graph_structure():
    data = np.array([
        ('A', 0, 0, 0, 0),
        ('B', 0, 0, 1, 0),
        ('C', 0, 1, 0, 0),
        ('D', 0, 1, 1, 0),
        ('E', 1, 0, 0, 1),
        ('F', 1, 0, 1, 1),
        ('G', 1, 1, 0, 1),
        ('H', 1, 1, 1, 1)
    ], dtype=[('node_id', 'U1'), ('x1', 'f4'), ('x2', 'f4'), ('x3', 'f4'), ('y', 'f4')])

    X = np.stack([data['x1'], data['x2'], data['x3']]).T
    Y = data['y']  # Extract the last column

    # Convert to PyTorch tensors
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)

    nodes = data['node_id'].tolist()
    links = [
        ('A', 'B'),
        ('B', 'C'),
        ('B', 'D'),
        ('B', 'F'),
        ('C', 'D'),
        ('C', 'E'),
        ('C', 'G'),
        ('D', 'E'),
        ('D', 'F'),
        ('D', 'A'),
        ('E', 'C'),
        ('E', 'F'),
        ('E', 'G'),
        ('E', 'H'),
        ('F', 'G'),
        ('F', 'A'),
        ('G', 'H'),
        ('G', 'A'),
        ('G', 'B'),
        ('H', 'A'),
        ('H', 'B'),
    ]
    return graph_structure(nodes=nodes, links=links, directed=False, device=device), X, Y


@pytest.mark.parametrize("interdependence_class", [graph_interdependence, multihop_graph_interdependence, pagerank_multihop_graph_interdependence])
def test_graph_interdependence(sample_graph_structure, interdependence_class):
    """
    Test basic graph interdependence and multihop graph interdependence.
    """
    graph, X, Y = sample_graph_structure
    interdep = interdependence_class(b=8, m=3, graph=graph, normalization=True, normalization_mode='row_column', device=device)

    A = interdep.calculate_A(device=device)
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (8, 8), "A shape mismatch"
    print(A, X, interdep(X, device=device), torch.matmul(A, X))

    # Test self dependence option
    interdep_self = interdependence_class(b=8, m=3, graph=graph, normalization=True, normalization_mode='row_column', self_dependence=True, device=device)
    A_self = interdep_self.calculate_A(device=device)
    print(A_self, X, interdep_self(X, device=device), torch.matmul(A, X))
    assert A_self.shape == (8, 8), "Self-dependence A shape mismatch"
    assert (A_self - A).sum() >= 0, "Self-dependence should modify the matrix"
