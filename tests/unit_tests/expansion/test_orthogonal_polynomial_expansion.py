# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################################################
# Test Orthogonal Polynomial based Expansion Function #
#######################################################

import pytest
import torch

# Assuming that the transformation classes have been defined as provided in the user input
from tinybig.expansion.orthogonal_polynomial_expansion import (
    hermite_expansion, laguerre_expansion, legendre_expansion, gegenbauer_expansion,
    bessel_expansion, reverse_bessel_expansion, fibonacci_expansion, lucas_expansion
)

# Test data for validation
@pytest.fixture
def test_data():
    # Create a random tensor of shape (batch_size, m)
    batch_size = 5
    m = 3
    x = torch.rand(batch_size, m)
    return x

# Test case for Hermite Expansion
def test_hermite_expansion(test_data):
    expander = hermite_expansion(d=3)
    result = expander.forward(test_data)
    assert result.shape == (test_data.shape[0], test_data.shape[1] * expander.d), "Hermite expansion shape mismatch"

# Test case for Laguerre Expansion
def test_laguerre_expansion(test_data):
    expander = laguerre_expansion(d=3, alpha=1.0)
    result = expander.forward(test_data)
    assert result.shape == (test_data.shape[0], test_data.shape[1] * expander.d), "Laguerre expansion shape mismatch"

# Test case for Legendre Expansion
def test_legendre_expansion(test_data):
    expander = legendre_expansion(d=3)
    result = expander.forward(test_data)
    assert result.shape == (test_data.shape[0], test_data.shape[1] * expander.d), "Legendre expansion shape mismatch"

# Test case for Gegenbauer Expansion
def test_gegenbauer_expansion(test_data):
    expander = gegenbauer_expansion(d=3, alpha=1.0)
    result = expander.forward(test_data)
    assert result.shape == (test_data.shape[0], test_data.shape[1] * expander.d), "Gegenbauer expansion shape mismatch"

# Test case for Bessel Expansion
def test_bessel_expansion(test_data):
    expander = bessel_expansion(d=3)
    result = expander.forward(test_data)
    assert result.shape == (test_data.shape[0], test_data.shape[1] * expander.d), "Bessel expansion shape mismatch"

# Test case for Reverse Bessel Expansion
def test_reverse_bessel_expansion(test_data):
    expander = reverse_bessel_expansion(d=3)
    result = expander.forward(test_data)
    assert result.shape == (test_data.shape[0], test_data.shape[1] * expander.d), "Reverse Bessel expansion shape mismatch"

# Test case for Fibonacci Expansion
def test_fibonacci_expansion(test_data):
    expander = fibonacci_expansion(d=3)
    result = expander.forward(test_data)
    assert result.shape == (test_data.shape[0], test_data.shape[1] * expander.d), "Fibonacci expansion shape mismatch"

# Test case for Lucas Expansion
def test_lucas_expansion(test_data):
    expander = lucas_expansion(d=3)
    result = expander.forward(test_data)
    assert result.shape == (test_data.shape[0], test_data.shape[1] * expander.d), "Lucas expansion shape mismatch"

# Execute the tests
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
