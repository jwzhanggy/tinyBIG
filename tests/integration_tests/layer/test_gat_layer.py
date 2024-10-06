# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Test Topological Interdependence #
###############################

import pytest
import torch
import numpy as np

from tinybig.koala.topology import graph as graph_structure
from tinybig.layer.graph_based_layers import gat_layer

device = 'mps'

@pytest.fixture
def sample_graph_structure():
    X = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
    ]
    Y = [0, 0, 0, 0, 1, 1, 1, 1]

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
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
    return graph_structure(nodes=nodes, links=links, directed=False), X, Y


def test_graph_interdependence(sample_graph_structure):
    """
    Test basic graph interdependence and multihop graph interdependence.
    """
    graph, X, Y = sample_graph_structure

    head_1 = gat_layer(
        width=2,
        channel_num=2,
        m=5, n=3,
        graph=graph,
        enable_bias=True,
        device=device
    )
    head_2 = gat_layer(
        width=2,
        channel_num=2,
        m=3, n=1,
        graph=graph,
        enable_bias=True,
        device=device
    )
    X = head_1(X, device=device)
    print(X.shape)
    X = head_2(X, device=device)
