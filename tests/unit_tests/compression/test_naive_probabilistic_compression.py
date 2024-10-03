# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################################################
# Test Naive Probabilistic based Compression Function #
#######################################################


import pytest
import torch
import numpy as np
from tinybig.koala.statistics import batch_mean
from tinybig.compression.probabilistic_compression import (
    naive_probabilistic_compression,
    naive_normal_probabilistic_compression,
    naive_cauchy_probabilistic_compression,
    naive_chi2_probabilistic_compression,
    naive_exponential_probabilistic_compression,
    naive_gamma_probabilistic_compression,
    naive_laplace_probabilistic_compression
)  # Adjust the import according to your file structure

# Mock implementations of the metric functions for testing purposes
def mock_batch_mean(x: torch.Tensor) -> torch.Tensor:
    return 3.0*x

@pytest.fixture
def setup_naive_probabilistic_compression():
    return naive_probabilistic_compression(k=2, metric=mock_batch_mean, require_normalization=True)  # Use mock mean

@pytest.fixture
def setup_naive_simply_sampling_probabilistic_compression():
    return naive_probabilistic_compression(k=2, metric=mock_batch_mean, simply_sampling=False, require_normalization=True)  # Use mock mean


def test_naive_probabilistic_initialization(setup_naive_probabilistic_compression):
    prob_comp = setup_naive_probabilistic_compression
    assert prob_comp.k == 2
    assert prob_comp.metric is not None


def test_naive_simply_sampling_probabilistic_initialization(setup_naive_simply_sampling_probabilistic_compression):
    prob_comp = setup_naive_simply_sampling_probabilistic_compression
    assert prob_comp.k == 2
    assert prob_comp.metric is not None

def test_naive_normal_probabilistic_compression_initialization():
    normal_comp = naive_normal_probabilistic_compression(k=2)
    assert normal_comp.k == 2
    assert isinstance(normal_comp.distribution_function, torch.distributions.normal.Normal)

def test_naive_cauchy_probabilistic_compression_initialization():
    cauchy_comp = naive_cauchy_probabilistic_compression(k=2)
    assert cauchy_comp.k == 2
    assert isinstance(cauchy_comp.distribution_function, torch.distributions.cauchy.Cauchy)

def test_naive_chi2_probabilistic_compression_initialization():
    chi2_comp = naive_chi2_probabilistic_compression(k=2)
    assert chi2_comp.k == 2
    assert isinstance(chi2_comp.distribution_function, torch.distributions.chi2.Chi2)

def test_naive_exponential_probabilistic_compression_initialization():
    exponential_comp = naive_exponential_probabilistic_compression(k=2)
    assert exponential_comp.k == 2
    assert isinstance(exponential_comp.distribution_function, torch.distributions.exponential.Exponential)

def test_naive_gamma_probabilistic_compression_initialization():
    gamma_comp = naive_gamma_probabilistic_compression(k=2)
    assert gamma_comp.k == 2
    assert isinstance(gamma_comp.distribution_function, torch.distributions.gamma.Gamma)

def test_naive_laplace_probabilistic_compression_initialization():
    laplace_comp = naive_laplace_probabilistic_compression(k=2)
    assert laplace_comp.k == 2
    assert isinstance(laplace_comp.distribution_function, torch.distributions.laplace.Laplace)

def test_naive_probabilistic_forward(setup_naive_probabilistic_compression):
    prob_comp = setup_naive_probabilistic_compression
    x = torch.tensor([[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]])
    result = prob_comp.forward(x)
    assert result.shape == (2, 2)  # Expecting (batch_size, d) -> (2, 2)

def test_naive_simply_sampling_probabilistic_forward(setup_naive_simply_sampling_probabilistic_compression):
    prob_comp = setup_naive_simply_sampling_probabilistic_compression
    x = torch.tensor([[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]])
    result = prob_comp.forward(x)
    assert result.shape == (2, 2)  # Expecting (batch_size, d) -> (2, 2)

def test_naive_normal_probabilistic_forward():
    normal_comp = naive_normal_probabilistic_compression(k=2)
    x = torch.tensor([[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]])
    result = normal_comp.forward(x)
    assert result.shape == (2, 2)  # Expecting (batch_size, d) -> (2, 2)

def test_naive_cauchy_probabilistic_forward():
    cauchy_comp = naive_cauchy_probabilistic_compression(k=2)
    x = torch.tensor([[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]])
    result = cauchy_comp.forward(x)
    assert result.shape == (2, 2)  # Expecting (batch_size, d) -> (2, 2)

def test_naive_chi2_probabilistic_forward():
    chi2_comp = naive_chi2_probabilistic_compression(k=2)
    x = torch.tensor([[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]])
    result = chi2_comp.forward(x)
    assert result.shape == (2, 2)  # Expecting (batch_size, d) -> (2, 2)

def test_naive_exponential_probabilistic_forward():
    exponential_comp = naive_exponential_probabilistic_compression(k=2)
    x = torch.tensor([[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]])
    result = exponential_comp.forward(x)
    assert result.shape == (2, 2)  # Expecting (batch_size, d) -> (2, 2)

def test_naive_gamma_probabilistic_forward():
    gamma_comp = naive_gamma_probabilistic_compression(k=2)
    x = torch.tensor([[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]])
    result = gamma_comp.forward(x)
    assert result.shape == (2, 2)  # Expecting (batch_size, d) -> (2, 2)

def test_naive_laplace_probabilistic_forward():
    laplace_comp = naive_laplace_probabilistic_compression(k=2)
    x = torch.tensor([[1.0, 2.0, 8.0], [3.0, 4.0, 9.0]])
    result = laplace_comp.forward(x)
    assert result.shape == (2, 2)  # Expecting (batch_size, d) -> (2, 2)

# Run the tests
if __name__ == "__main__":
    pytest.main()
