# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Test Hybrid Interdependence #
###############################

import pytest
import torch
import numpy as np
from tinybig.interdependence.topological_interdependence import (
    graph_interdependence,
    multihop_graph_interdependence,
    pagerank_multihop_graph_interdependence
)
from tinybig.interdependence.parameterized_bilinear_interdependence import (
    parameterized_bilinear_interdependence,
    lowrank_parameterized_bilinear_interdependence,
    hm_parameterized_bilinear_interdependence,
    lphm_parameterized_bilinear_interdependence,
    dual_lphm_parameterized_bilinear_interdependence,
    random_matrix_adaption_parameterized_bilinear_interdependence
)
from tinybig.interdependence import hybrid_interdependence
from tinybig.koala.topology import graph as graph_structure
from tinybig.fusion.metric_fusion import prod_fusion

device = 'mps'

@pytest.fixture
def sample_graph_structure():
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
    return graph_structure(nodes=nodes, links=links, directed=False, device=device), X, y


@pytest.fixture
def interdependence_factory():
    def _create_interdependence(interdependence_class, b, m, **kwargs):
        # Instantiate the interdependence class with b, m, and pass other parameters
        interdep = interdependence_class(b=b, m=m, device=device, **kwargs)
        # Dynamically calculate l using the calculate_l method
        l = interdep.calculate_l()
        return interdep, l

    return _create_interdependence

@pytest.mark.parametrize("bilinear_interdependence_class, extra_params", [
    (parameterized_bilinear_interdependence, {"interdependence_type": "instance"}),
    (lowrank_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "r": 2}),
    (hm_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "p": 1, "q": 1}),
    (lphm_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "r": 2, "p": 1, "q": 1}),
    (dual_lphm_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "r": 2, "p": 1, "q": 1}),
    (random_matrix_adaption_parameterized_bilinear_interdependence, {"interdependence_type": "instance", "r": 2}),
])
@pytest.mark.parametrize("graph_interdependence_class", [graph_interdependence, multihop_graph_interdependence, pagerank_multihop_graph_interdependence])
def test_graph_interdependence(sample_graph_structure, interdependence_factory, bilinear_interdependence_class, extra_params, graph_interdependence_class):
    """
    Test basic graph interdependence and multihop graph interdependence.
    """
    graph, X, y = sample_graph_structure
    graph_interdep = graph_interdependence_class(b=8, m=3, graph=graph, interdependence_type='instance', normalization=False, device=device)
    bilinear_interdep, _ = interdependence_factory(bilinear_interdependence_class, b=8, m=3, **extra_params)

    hybrid_interdep = hybrid_interdependence(
        b=8, m=3,
        interdependence_type='instance',
        interdependence_functions=[
            graph_interdep, bilinear_interdep
        ],
        fusion_function=prod_fusion(dims=[8]*2)
    )
    l = hybrid_interdep.calculate_l()
    #print('l', l)
    w = torch.randn((1, l), device=device)
    A_hybrid = hybrid_interdep.calculate_A(x=X.T, w=w, device=device)
    assert A_hybrid.shape == (8, 8), "A shape mismatch"
    #print(A_hybrid, X, graph_interdep(X, device=device), torch.matmul(A_hybrid.T, X))

