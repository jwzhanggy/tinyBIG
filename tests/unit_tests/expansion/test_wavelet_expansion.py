# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################################
# Test Wavelets based Expansion Function #
##########################################

import pytest
import torch
from tinybig.expansion.wavelet_expansion import (
    harr_wavelet_expansion,
    beta_wavelet_expansion,
    shannon_wavelet_expansion,
    ricker_wavelet_expansion,
    dog_wavelet_expansion,
    meyer_wavelet_expansion
)

# Test Data
@pytest.fixture
def test_tensor():
    return torch.randn(5, 10)  # 5 samples, 10 features

# Test Harr Wavelet Expansion
def test_harr_wavelet_expansion(test_tensor):
    expansion = harr_wavelet_expansion(d=2, s=2, t=2)
    result = expansion.forward(test_tensor)
    assert result.shape == (5, expansion.calculate_D(m=10)), "Harr Wavelet Expansion output shape is incorrect."

# Test Beta Wavelet Expansion
def test_beta_wavelet_expansion(test_tensor):
    expansion = beta_wavelet_expansion(d=2, s=2, t=2, alpha=2.0, beta=2.0)
    result = expansion.forward(test_tensor)
    assert result.shape == (5, expansion.calculate_D(m=10)), "Beta Wavelet Expansion output shape is incorrect."

# Test Shannon Wavelet Expansion
def test_shannon_wavelet_expansion(test_tensor):
    expansion = shannon_wavelet_expansion(d=2, s=2, t=2)
    result = expansion.forward(test_tensor)
    assert result.shape == (5, expansion.calculate_D(m=10)), "Shannon Wavelet Expansion output shape is incorrect."

# Test Ricker Wavelet Expansion
def test_ricker_wavelet_expansion(test_tensor):
    expansion = ricker_wavelet_expansion(d=2, s=2, t=2, sigma=1.0)
    result = expansion.forward(test_tensor)
    assert result.shape == (5, expansion.calculate_D(m=10)), "Ricker Wavelet Expansion output shape is incorrect."

# Test Difference of Gaussians Wavelet Expansion
def test_dog_wavelet_expansion(test_tensor):
    expansion = dog_wavelet_expansion(d=2, s=2, t=2, sigma_1=1.0, sigma_2=2.0)
    result = expansion.forward(test_tensor)
    assert result.shape == (5, expansion.calculate_D(m=10)), "Difference of Gaussians Wavelet Expansion output shape is incorrect."

# Test Meyer Wavelet Expansion
def test_meyer_wavelet_expansion(test_tensor):
    expansion = meyer_wavelet_expansion(d=2, s=2, t=2)
    result = expansion.forward(test_tensor)
    assert result.shape == (5, expansion.calculate_D(m=10)), "Meyer Wavelet Expansion output shape is incorrect."

# Test ValueError for invalid parameters
def test_invalid_parameters():
    with pytest.raises(ValueError):
        _ = harr_wavelet_expansion(a=0.5, b=1.0)  # a must be > 1
    with pytest.raises(ValueError):
        _ = beta_wavelet_expansion(alpha=0.5, beta=1.0)  # alpha and beta must be >= 1
    with pytest.raises(ValueError):
        _ = dog_wavelet_expansion(sigma_1=-1, sigma_2=1)  # sigma_1 must be >= 0
    with pytest.raises(ValueError):
        _ = ricker_wavelet_expansion(sigma=-1)  # sigma must be >= 0

if __name__ == "__main__":
    pytest.main()
