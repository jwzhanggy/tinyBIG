# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################
# Test Random Matrix Reconciliation #
#####################################

import pytest
import torch
from tinybig.reconciliation import random_matrix_adaption_reconciliation, random_matrix_hypernet_reconciliation

device = 'mps'
# Test Data
@pytest.fixture
def test_parameters():
    n, D, r, l = 3, 4, 2, 5
    # Create a parameter tensor with shape (n + r, D)
    w1 = torch.randn(1, n + r, device=device)
    w2 = torch.randn(1, l, device=device)
    return n, D, r, l, w1, w2

# Test Random Matrix Adaption Reconciliation
def test_random_matrix_adaption_reconciliation(test_parameters):
    n, D, r, l, w1, w2 = test_parameters
    reconciliation = random_matrix_adaption_reconciliation(r=r)

    # Test forward method
    result = reconciliation.forward(n=n, D=D, w=w1, device=device)
    expected_shape = (n, D)
    assert result.shape == expected_shape, "Random Matrix Adaption Reconciliation output shape is incorrect."

    # Test that the random matrices A and B are initialized correctly
    assert reconciliation.A.shape == (n, r), "Matrix A shape is incorrect."
    assert reconciliation.B.shape == (D, r), "Matrix B shape is incorrect."

# Test Random Matrix Hypernet Reconciliation
def test_random_matrix_hypernet_reconciliation(test_parameters):
    n, D, r, l, w1, w2 = test_parameters
    reconciliation = random_matrix_hypernet_reconciliation(r=r, l=5, hidden_dim=10)

    # Test forward method
    result = reconciliation.forward(n=n, D=D, w=w2, device=device)
    expected_shape = (n, D)
    assert result.shape == expected_shape, "Random Matrix Hypernet Reconciliation output shape is incorrect."

    # Test that the random matrices P, Q, S, and T are initialized correctly
    assert reconciliation.P.shape == (reconciliation.l, r), "Matrix P shape is incorrect."
    assert reconciliation.Q.shape == (reconciliation.hidden_dim, r), "Matrix Q shape is incorrect."
    assert reconciliation.S.shape == (reconciliation.hidden_dim, r), "Matrix S shape is incorrect."
    assert reconciliation.T.shape == (n * D, r), "Matrix T shape is incorrect."

# Test ValueError for invalid parameters
def test_invalid_parameters():
    with pytest.raises(AssertionError):
        # Wrong shape of w
        w = torch.randn(5, 3)  # This should fail since it doesn't match the expected dimensions
        reconciliation = random_matrix_adaption_reconciliation()
        reconciliation.forward(n=3, D=4, w=w)

    with pytest.raises(AssertionError):
        # Wrong shape of w
        w = torch.randn(5, 3)  # This should fail since it doesn't match the expected dimensions
        reconciliation = random_matrix_hypernet_reconciliation()
        reconciliation.forward(n=3, D=4, w=w)

if __name__ == "__main__":
    pytest.main()
