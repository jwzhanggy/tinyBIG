# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# Test Recurrent Head #
#######################

import pytest
import torch
import numpy as np
from tinybig.head import recurrent_head

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

def test_graph_interdependence(sample_graph_structure):
    """
    Test basic graph interdependence and multihop graph interdependence.
    """
    chain_length, X, Y = sample_graph_structure
    head = recurrent_head(
        m=3, n=2,
        chain_length=chain_length,
        device=device
    )
    print(head.instance_interdependence.calculate_A())
    print('X', X, 'Xi_X', head(X, device=device))
