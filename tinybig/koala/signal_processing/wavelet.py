# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################
# Wavelet Analysis #
####################

from abc import abstractmethod

import math
from scipy.special import beta
import torch


class discrete_wavelet(object):
    def __init__(self, name: str = 'discrete_wavelet', a: float = 1.0, b: float = 1.0, *args, **kwargs):
        if a < 1 or b < 0:
            raise ValueError('a must be > 1 and b mush > 0...')

        self.name = name
        self.a = a
        self.b = b

    @abstractmethod
    def psi(self, tau: torch.Tensor):
        pass

    def forward(self, x: torch.Tensor, s: int, t: int):
        tau = x/(self.a**s) - t*self.b
        return 1.0/math.sqrt(self.a**s) * self.psi(tau=tau)

    def __call__(self, x: torch.Tensor, s: int, t: int):
        return self.forward(x=x, s=s, t=t)


class harr_wavelet(discrete_wavelet):
    def __init__(self, name: str = 'harr_wavelet', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def psi(self, tau: torch.Tensor):
        result = torch.zeros_like(tau)
        result[(tau >= 0) & (tau < 0.5)] = 1
        result[(tau >= 0.5) & (tau < 1)] = -1
        return result


class beta_wavelet(discrete_wavelet):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, name: str = 'beta_wavelet', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        if self.alpha < 1.0 or self.beta < 1.0:
            raise ValueError('alpha and beta must be >= 1.')

    def psi(self, tau: torch.Tensor):
        if not torch.all((tau >= 0) & (tau <= 1)):
            tau = torch.sigmoid(tau)
        assert torch.all((tau >= 0) & (tau <= 1))
        beta_coeff = 1.0/beta(self.alpha, self.beta)
        return beta_coeff * tau**(self.alpha - 1) * (1.0 - tau)**(self.beta - 1.0)


class shannon_wavelet(discrete_wavelet):
    def __init__(self, name: str = 'shannon_wavelet', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def psi(self, tau: torch.Tensor):
        return (torch.sin(2*torch.pi*tau) - torch.sin(torch.pi*tau))/(torch.pi*tau)


class ricker_wavelet(discrete_wavelet):
    def __init__(self, sigma: float = 1.0, name: str = 'ricker_wavelet', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        if sigma < 0.0:
            raise ValueError('sigma must be >= 0.')
        self.sigma = sigma

    def psi(self, tau: torch.Tensor):
        term1 = 2.0*(1.0-(tau/self.sigma)**2)/(math.sqrt(3*self.sigma)*(torch.pi**0.25))
        term2 = torch.exp(-tau**2/(2*self.sigma**2))
        return term1*term2


class dog_wavelet(discrete_wavelet):
    def __init__(self, sigma_1: float = 1.0, sigma_2: float = 2.0, name: str = 'difference_of_Gaussians_wavelet', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        if sigma_1 < 0.0 or sigma_2 < 0.0:
            raise ValueError('sigma_1 and sigma_2 must be >= 0.')
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

    def psi(self, tau: torch.Tensor):
        gauss1 = torch.exp(-0.5 * (tau / self.sigma_1) ** 2) / (math.sqrt(2 * torch.pi * self.sigma_1 ** 2))
        gauss2 = torch.exp(-0.5 * (tau / self.sigma_2) ** 2) / (math.sqrt(2 * torch.pi * self.sigma_2 ** 2))
        return gauss1 - gauss2


class meyer_wavelet(discrete_wavelet):
    def __init__(self, name: str = 'meyer_wavelet', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def psi(self, tau: torch.Tensor):
        result = torch.zeros_like(tau)

        zero_mask = (tau == 0)
        result[zero_mask] = 2.0/3.0 + 4.0/(3.0 * torch.pi)

        nonzero_mask = ~zero_mask
        t_nonzero = tau[nonzero_mask]
        result[nonzero_mask] = (
            (torch.sin((2.0 * torch.pi / 3.0) * t_nonzero) +
             4.0/3.0 * t_nonzero * torch.cos((4.0 * torch.pi / 3.0) * t_nonzero)) /
            (torch.pi * t_nonzero - (16.0 * torch.pi / 9.0) * t_nonzero ** 3)
        )
        return result


