# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Test Topological Interdependence #
###############################

import pytest
import torch
from tinybig.interdependence.topological_interdependence import (
    chain_interdependence,
    multihop_chain_interdependence,
    inverse_approx_multihop_chain_interdependence,
    exponential_approx_multihop_chain_interdependence,
    graph_interdependence,
    multihop_graph_interdependence,
    pagerank_multihop_graph_interdependence
)
from tinybig.koala.topology import chain as chain_structure
from tinybig.koala.topology import graph as graph_structure

@pytest.fixture
def sample_chain_structure():
    return chain_structure(length=5, bi_directional=False)

@pytest.fixture
def sample_graph_structure():
    return graph_structure(nodes=[0, 1, 2, 3, 4], links=[(0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (3, 2), (3, 4)], directed=True)


@pytest.mark.parametrize("interdependence_class", [chain_interdependence, multihop_chain_interdependence])
def test_chain_interdependence(sample_chain_structure, interdependence_class):
    """
    Test basic chain interdependence and multihop chain interdependence
    """
    chain = sample_chain_structure
    interdep = interdependence_class(b=3, m=5, chain=chain, interdependence_type='attribute', device='cpu')

    A = interdep.calculate_A()
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (5, 5), "A shape mismatch"

    # Test self dependence option
    interdep_self = interdependence_class(b=3, m=5, chain=chain, self_dependence=True, interdependence_type='attribute', device='cpu')
    A_self = interdep_self.calculate_A()
    assert A_self.shape == (5, 5), "Self-dependence A shape mismatch"
    assert (A_self - A).sum() > 0, "Self-dependence should modify the matrix"


@pytest.mark.parametrize("interdependence_class", [multihop_chain_interdependence])
@pytest.mark.parametrize("accumulative", [True, False])
def test_multihop_chain_interdependence(sample_chain_structure, interdependence_class, accumulative):
    """
    Test multihop chain interdependence with accumulation option.
    """
    chain = sample_chain_structure
    interdep = interdependence_class(b=5, m=5, chain=chain, h=2, accumulative=accumulative, device='cpu')

    A = interdep.calculate_A()
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (5, 5), "A shape mismatch"

    if accumulative:
        assert (A - torch.eye(5)).sum() > 0, "Accumulation should modify the matrix"


def test_inverse_approx_multihop_chain_interdependence(sample_chain_structure):
    """
    Test approximate multihop chain interdependence with reciprocal and exponential approximations.
    """
    chain = sample_chain_structure
    interdep = inverse_approx_multihop_chain_interdependence(
        b=5, m=5, chain=chain, device='cpu'
    )

    A = interdep.calculate_A()
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (5, 5), "A shape mismatch"


def test_exponential_approx_multihop_chain_interdependence(sample_chain_structure):
    """
    Test approximate multihop chain interdependence with reciprocal and exponential approximations.
    """
    chain = sample_chain_structure
    interdep = exponential_approx_multihop_chain_interdependence(
        b=5, m=5, chain=chain, device='cpu'
    )

    A = interdep.calculate_A()
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (5, 5), "A shape mismatch"


@pytest.mark.parametrize("interdependence_class", [graph_interdependence, multihop_graph_interdependence])
def test_graph_interdependence(sample_graph_structure, interdependence_class):
    """
    Test basic graph interdependence and multihop graph interdependence.
    """
    graph = sample_graph_structure
    interdep = interdependence_class(b=5, m=5, graph=graph, device='cpu')

    A = interdep.calculate_A()
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (5, 5), "A shape mismatch"

    # Test self dependence option
    interdep_self = interdependence_class(b=5, m=5, graph=graph, self_dependence=True, device='cpu')
    A_self = interdep_self.calculate_A()
    assert A_self.shape == (5, 5), "Self-dependence A shape mismatch"
    assert (A_self - A).sum() > 0, "Self-dependence should modify the matrix"


def test_pagerank_multihop_graph_interdependence(sample_graph_structure):
    """
    Test pagerank multihop graph interdependence.
    """
    graph = sample_graph_structure
    interdep = pagerank_multihop_graph_interdependence(b=5, m=5, graph=graph, c=0.15, device='cpu')

    A = interdep.calculate_A()
    assert A is not None, "Matrix A should not be None"
    assert A.shape == (5, 5), "A shape mismatch"


def test_exceptions():
    """
    Test exceptions for invalid input.
    """
    with pytest.raises(ValueError):
        chain_interdependence(b=5, m=5, chain=None)

    with pytest.raises(ValueError):
        inverse_approx_multihop_chain_interdependence(b=5, m=5, chain=None)

    with pytest.raises(ValueError):
        graph_interdependence(b=5, m=5, graph=None)
