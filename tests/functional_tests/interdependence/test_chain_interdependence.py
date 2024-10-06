# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Test Chain Interdependence #
###############################

import pytest
import torch
import numpy as np
from tinybig.interdependence.topological_interdependence import (
    chain_interdependence,
    multihop_chain_interdependence,
    inverse_approx_multihop_chain_interdependence,
    exponential_approx_multihop_chain_interdependence
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
    ], dtype=[('node_id', 'U1'), ('x1', 'f4'), ('x2', 'f4'), ('x3', 'f4'), ('y', 'f4')])

    X = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    ).to(device)
    y = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1]
    ).to(device)

    return 8, X, y

@pytest.mark.parametrize("interdependence_class", [chain_interdependence, multihop_chain_interdependence, inverse_approx_multihop_chain_interdependence, exponential_approx_multihop_chain_interdependence])
def test_graph_interdependence(sample_graph_structure, interdependence_class):
    """
    Test basic graph interdependence and multihop graph interdependence.
    """
    chain_length, X, Y = sample_graph_structure
    interdep = interdependence_class(b=8, m=3, length=chain_length,  bi_directional=False, device=device)

    A = interdep.calculate_A(device=device)
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (8, 8), "A shape mismatch"
    print(interdependence_class, 'A', A, 'X', X, 'Xi_X', interdep(X, device=device), 'matmul', torch.matmul(A.T, X))

    # Test bi-directional option
    interdep_self = interdependence_class(b=8, m=3, length=chain_length, bi_directional=True, device=device)
    A_bi = interdep_self.calculate_A(device=device)
    print('A_bi', A_bi, 'X', X, 'Xi_X', interdep_self(X, device=device), 'matmul', torch.matmul(A_bi.T, X))
    assert A_bi.shape == (8, 8), "Self-dependence A shape mismatch"
    #print(interdependence_class, 'A', A, 'A_bi', A_bi)
