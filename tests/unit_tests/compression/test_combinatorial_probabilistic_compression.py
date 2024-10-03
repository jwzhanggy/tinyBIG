# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################################################
# Test Combinatorial Probabilistic based Compression Function #
###############################################################

import pytest
import torch
from tinybig.compression.combinatorial_compression import combinatorial_compression  # Adjust the import according to your file structure

@pytest.fixture
def setup_combinatorial_compression():
    return combinatorial_compression(d=2, k=1, simply_sampling=True)  # Example setup

def test_combinatorial_initialization(setup_combinatorial_compression):
    comb_comp = setup_combinatorial_compression
    assert comb_comp.d == 2
    assert comb_comp.k == 1
    assert comb_comp.simply_sampling is True
    assert comb_comp.with_replacement is False

def test_calculate_D_simply_sampling(setup_combinatorial_compression):
    comb_comp = setup_combinatorial_compression
    assert comb_comp.calculate_D(m=3) == 3  # For simply_sampling, k*(1 + 2) = 1*1 + 1*2 = 3

def test_calculate_D_non_simply_sampling():
    comb_comp = combinatorial_compression(d=2, k=2, simply_sampling=False)
    assert comb_comp.calculate_D(m=3) == 4  # For non-simply_sampling, d*k = 2*2 = 4

def test_calculate_weights(setup_combinatorial_compression):
    comb_comp = setup_combinatorial_compression
    x = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    weights = comb_comp.calculate_weights(x, r=2)  # Using first r = 1
    assert weights.shape == (2,)  # Expecting weights shape to match input

def test_random_combinations(setup_combinatorial_compression):
    comb_comp = setup_combinatorial_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    compression = comb_comp.random_combinations(x, r=2)  # Testing with r=2
    assert compression.shape == (2, 1, 2) # Expecting compression shape to match combinations

def test_forward(setup_combinatorial_compression):
    comb_comp = setup_combinatorial_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    compression = comb_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == comb_comp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_forward_with_replacement():
    comb_comp = combinatorial_compression(d=2, k=1, simply_sampling=True, with_replacement=True)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    compression = comb_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == comb_comp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_forward_log_prob():
    comb_comp = combinatorial_compression(d=2, k=1, simply_sampling=True, log_prob=True)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    compression = comb_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == comb_comp.calculate_D(m=x.shape[1])  # Should match calculated D

# Run the tests
if __name__ == "__main__":
    pytest.main()


import pytest
import torch
from tinybig.compression.combinatorial_compression import combinatorial_probabilistic_compression  # Adjust the import according to your file structure

@pytest.fixture
def setup_combinatorial_probabilistic_compression():
    return combinatorial_probabilistic_compression(d=2, k=1)  # Example setup

def test_initialization(setup_combinatorial_probabilistic_compression):
    comb_prob_comp = setup_combinatorial_probabilistic_compression
    assert comb_prob_comp.d == 2
    assert comb_prob_comp.k == 1
    assert comb_prob_comp.simply_sampling is False
    assert comb_prob_comp.with_replacement is False
    assert comb_prob_comp.require_normalization is True
    assert comb_prob_comp.log_prob is True

def test_calculate_D(setup_combinatorial_probabilistic_compression):
    comb_prob_comp = setup_combinatorial_probabilistic_compression
    assert comb_prob_comp.calculate_D(m=3) == 2  # Should return d * k = 2 * 1 = 2

def test_calculate_weights(setup_combinatorial_probabilistic_compression):
    comb_prob_comp = setup_combinatorial_probabilistic_compression
    x = torch.tensor([[1.0], [4.0]])
    weights = comb_prob_comp.calculate_weights(x, r=1)  # Using first r = 1
    assert weights.shape == (2,)  # Expecting weights shape to match input

def test_forward(setup_combinatorial_probabilistic_compression):
    comb_prob_comp = setup_combinatorial_probabilistic_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    compression = comb_prob_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == comb_prob_comp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_forward_with_replacement():
    comb_prob_comp = combinatorial_probabilistic_compression(d=2, k=1, with_replacement=True)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    compression = comb_prob_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == comb_prob_comp.calculate_D(m=x.shape[1])  # Should match calculated D

def test_forward_log_prob():
    comb_prob_comp = combinatorial_probabilistic_compression(d=2, k=1)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    compression = comb_prob_comp.forward(x)
    assert compression.shape[0] == 2  # Should match batch size
    assert compression.shape[1] == comb_prob_comp.calculate_D(m=x.shape[1])  # Should match calculated D

# Run the tests
if __name__ == "__main__":
    pytest.main()
