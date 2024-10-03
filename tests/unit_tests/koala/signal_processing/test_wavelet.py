# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################################
# Test Wavelet in koala.signal_processing #
###########################################

import pytest
import torch
from tinybig.koala.signal_processing.wavelet import harr_wavelet, beta_wavelet, shannon_wavelet, ricker_wavelet, dog_wavelet, meyer_wavelet, discrete_wavelet


# Utility function to generate random data
def generate_tensor_data(size=100):
    return torch.rand(size)


# Base tests for discrete_wavelet
class Test_Discrete_Wavelet_Base:

    def test_initialization(self):
        with pytest.raises(ValueError):
            dw = discrete_wavelet(a=0.5, b=-1)

    def test_forward_call(self):
        wavelet = harr_wavelet()
        x = generate_tensor_data()
        result = wavelet(x=x, s=1, t=1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

# Test for Harr Wavelet
class Test_Harr_Wavelet:

    def test_harr_psi(self):
        wavelet = harr_wavelet()
        tau = torch.tensor([0.25, 0.75, 1.0, 1.5])
        psi_vals = wavelet.psi(tau=tau)
        expected = torch.tensor([1.0, -1.0, 0.0, 0.0])
        assert torch.equal(psi_vals, expected)

    def test_harr_forward(self):
        wavelet = harr_wavelet()
        x = generate_tensor_data()
        result = wavelet(x=x, s=2, t=1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape


# Test for Beta Wavelet
class Test_Beta_Wavelet:

    def test_beta_wavelet_initialization(self):
        with pytest.raises(AssertionError):
            wavelet = beta_wavelet(alpha=0.5, beta=0.5)  # Invalid alpha, beta

    def test_beta_wavelet_psi(self):
        wavelet = beta_wavelet(alpha=2.0, beta=2.0)
        tau = torch.linspace(0, 1, 10)
        psi_vals = wavelet.psi(tau=tau)
        assert psi_vals.shape == tau.shape

    def test_beta_wavelet_forward(self):
        wavelet = beta_wavelet(alpha=2.0, beta=2.0)
        x = generate_tensor_data()
        result = wavelet(x=x, s=1, t=1)
        assert isinstance(result, torch.Tensor)


# Test for Shannon Wavelet
class Test_Shannon_Wavelet:

    def test_shannon_wavelet_psi(self):
        wavelet = shannon_wavelet()
        tau = torch.tensor([0.25, 0.5, 1.0, 1.5])
        psi_vals = wavelet.psi(tau=tau)
        assert psi_vals.shape == tau.shape

    def test_shannon_wavelet_forward(self):
        wavelet = shannon_wavelet()
        x = generate_tensor_data()
        result = wavelet(x=x, s=2, t=1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape


# Test for Ricker Wavelet
class Test_Ricker_Wavelet:

    def test_ricker_wavelet_initialization(self):
        wavelet = ricker_wavelet(sigma=1.0)
        assert wavelet.sigma == 1.0

    def test_ricker_wavelet_psi(self):
        wavelet = ricker_wavelet(sigma=1.0)
        tau = torch.tensor([0.0, 0.5, 1.0, 1.5])
        psi_vals = wavelet.psi(tau=tau)
        assert psi_vals.shape == tau.shape

    def test_ricker_wavelet_forward(self):
        wavelet = ricker_wavelet(sigma=1.0)
        x = generate_tensor_data()
        result = wavelet(x=x, s=2, t=1)
        assert isinstance(result, torch.Tensor)


# Test for DOG Wavelet
class Test_Dog_Wavelet:

    def test_dog_wavelet_initialization(self):
        with pytest.raises(AssertionError):
            wavelet = dog_wavelet(sigma_1=-1.0, sigma_2=-1.0)  # Invalid sigmas

    def test_dog_wavelet_psi(self):
        wavelet = dog_wavelet(sigma_1=1.0, sigma_2=2.0)
        tau = torch.tensor([0.0, 0.5, 1.0, 1.5])
        psi_vals = wavelet.psi(tau=tau)
        assert psi_vals.shape == tau.shape

    def test_dog_wavelet_forward(self):
        wavelet = dog_wavelet(sigma_1=1.0, sigma_2=2.0)
        x = generate_tensor_data()
        result = wavelet(x=x, s=1, t=1)
        assert isinstance(result, torch.Tensor)


# Test for Meyer Wavelet
class Test_Meyer_Wavelet:

    def test_meyer_wavelet_psi(self):
        wavelet = meyer_wavelet()
        tau = torch.tensor([0.0, 0.5, 1.0, 1.5])
        psi_vals = wavelet.psi(tau=tau)
        assert psi_vals.shape == tau.shape

    def test_meyer_wavelet_forward(self):
        wavelet = meyer_wavelet()
        x = generate_tensor_data()
        result = wavelet(x=x, s=1, t=1)
        assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("wavelet_class", [harr_wavelet, beta_wavelet, shannon_wavelet, ricker_wavelet, dog_wavelet, meyer_wavelet])
def test_parametrized_wavelet_forward(wavelet_class):
    wavelet = wavelet_class()
    x = generate_tensor_data()
    result = wavelet(x=x, s=2, t=1)
    assert isinstance(result, torch.Tensor)
    assert result.shape == x.shape
