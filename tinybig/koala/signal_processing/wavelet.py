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
    """
    Base class for discrete wavelet functions.

    This class provides a framework for implementing various discrete wavelet functions with scaling and translation parameters.

    Attributes
    ----------
    name : str
        The name of the wavelet.
    a : float
        The scaling parameter, must be > 1.
    b : float
        The translation parameter, must be > 0.

    Methods
    -------
    psi(tau: torch.Tensor)
        Abstract method for wavelet function definition.
    forward(x: torch.Tensor, s: int, t: int)
        Apply the wavelet transformation to the input signal.
    __call__(x: torch.Tensor, s: int, t: int)
        Perform wavelet transformation via call.
    """
    def __init__(self, name: str = 'discrete_wavelet', a: float = 1.0, b: float = 1.0, *args, **kwargs):
        """
        Initialize the discrete wavelet class.

        Parameters
        ----------
        name : str, optional
            Name of the wavelet. Default is 'discrete_wavelet'.
        a : float, optional
            Scaling parameter. Must be > 1. Default is 1.0.
        b : float, optional
            Translation parameter. Must be > 0. Default is 1.0.
        *args, **kwargs
            Additional parameters.
        """
        if a < 1 or b < 0:
            raise ValueError('a must be > 1 and b mush > 0...')

        self.name = name
        self.a = a
        self.b = b

    @abstractmethod
    def psi(self, tau: torch.Tensor):
        """
        Abstract method for wavelet function definition.

        Parameters
        ----------
        tau : torch.Tensor
            Transformed input values.

        Returns
        -------
        torch.Tensor
            Wavelet values for the input.
        """
        pass

    def forward(self, x: torch.Tensor, s: int, t: int):
        """
        Apply the wavelet transformation to the input signal.

        Parameters
        ----------
        x : torch.Tensor
            Input signal.
        s : int
            Scaling factor.
        t : int
            Translation factor.

        Returns
        -------
        torch.Tensor
            Transformed signal.
        """
        tau = x/(self.a**s) - t*self.b
        return 1.0/math.sqrt(self.a**s) * self.psi(tau=tau)

    def __call__(self, x: torch.Tensor, s: int, t: int):
        """
        Perform wavelet transformation via call.

        Parameters
        ----------
        x : torch.Tensor
            Input signal.
        s : int
            Scaling factor.
        t : int
            Translation factor.

        Returns
        -------
        torch.Tensor
            Transformed signal.
        """
        return self.forward(x=x, s=s, t=t)


class harr_wavelet(discrete_wavelet):
    """
    Haar wavelet implementation.

    This wavelet is piecewise constant with values 1 or -1 over specific intervals.
    """
    def __init__(self, name: str = 'harr_wavelet', *args, **kwargs):
        """
        Initialize the Haar wavelet.

        Parameters
        ----------
        name : str, optional
            Name of the wavelet. Default is 'harr_wavelet'.
        *args, **kwargs
            Additional parameters.
        """
        super().__init__(name=name, *args, **kwargs)

    def psi(self, tau: torch.Tensor):
        """
        Define the Haar wavelet function.

        Parameters
        ----------
        tau : torch.Tensor
            Transformed input values.

        Returns
        -------
        torch.Tensor
            Wavelet values for the input.
        """
        result = torch.zeros_like(tau)
        result[(tau >= 0) & (tau < 0.5)] = 1
        result[(tau >= 0.5) & (tau < 1)] = -1
        return result


class beta_wavelet(discrete_wavelet):
    """
    Beta wavelet implementation.

    This wavelet uses the beta distribution for its wavelet function.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, name: str = 'beta_wavelet', *args, **kwargs):
        """
        Initialize the Beta wavelet.

        Parameters
        ----------
        alpha : float, optional
            Alpha parameter of the beta distribution. Must be >= 1. Default is 1.0.
        beta : float, optional
            Beta parameter of the beta distribution. Must be >= 1. Default is 1.0.
        name : str, optional
            Name of the wavelet. Default is 'beta_wavelet'.
        *args, **kwargs
            Additional parameters.
        """
        super().__init__(name=name, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        if self.alpha < 1.0 or self.beta < 1.0:
            raise ValueError('alpha and beta must be >= 1.')

    def psi(self, tau: torch.Tensor):
        """
        Define the Beta wavelet function.

        Parameters
        ----------
        tau : torch.Tensor
            Transformed input values.

        Returns
        -------
        torch.Tensor
            Wavelet values for the input.
        """
        if not torch.all((tau >= 0) & (tau <= 1)):
            tau = torch.sigmoid(tau)
        assert torch.all((tau >= 0) & (tau <= 1))
        beta_coeff = 1.0/beta(self.alpha, self.beta)
        return beta_coeff * tau**(self.alpha - 1) * (1.0 - tau)**(self.beta - 1.0)


class shannon_wavelet(discrete_wavelet):
    """
    Shannon wavelet implementation.

    This wavelet uses sinusoidal functions for its wavelet definition.
    """
    def __init__(self, name: str = 'shannon_wavelet', *args, **kwargs):
        """
        Initialize the Shannon wavelet.

        Parameters
        ----------
        name : str, optional
            Name of the wavelet. Default is 'shannon_wavelet'.
        *args, **kwargs
            Additional parameters.
        """
        super().__init__(name=name, *args, **kwargs)

    def psi(self, tau: torch.Tensor):
        """
        Define the Shannon wavelet function.

        Parameters
        ----------
        tau : torch.Tensor
            Transformed input values.

        Returns
        -------
        torch.Tensor
            Wavelet values for the input.
        """
        return (torch.sin(2*torch.pi*tau) - torch.sin(torch.pi*tau))/(torch.pi*tau)


class ricker_wavelet(discrete_wavelet):
    """
    Ricker wavelet implementation, also known as the "Mexican hat wavelet."
    """
    def __init__(self, sigma: float = 1.0, name: str = 'ricker_wavelet', *args, **kwargs):
        """
        Initialize the Ricker wavelet.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of the Gaussian envelope. Must be >= 0. Default is 1.0.
        name : str, optional
            Name of the wavelet. Default is 'ricker_wavelet'.
        *args, **kwargs
            Additional parameters.
        """
        super().__init__(name=name, *args, **kwargs)
        if sigma < 0.0:
            raise ValueError('sigma must be >= 0.')
        self.sigma = sigma

    def psi(self, tau: torch.Tensor):
        """
        Define the Ricker wavelet function.

        Parameters
        ----------
        tau : torch.Tensor
            Transformed input values.

        Returns
        -------
        torch.Tensor
            Wavelet values for the input.
        """
        term1 = 2.0*(1.0-(tau/self.sigma)**2)/(math.sqrt(3*self.sigma)*(torch.pi**0.25))
        term2 = torch.exp(-tau**2/(2*self.sigma**2))
        return term1*term2


class dog_wavelet(discrete_wavelet):
    """
    Difference of Gaussians (DoG) wavelet implementation.
    """
    def __init__(self, sigma_1: float = 1.0, sigma_2: float = 2.0, name: str = 'difference_of_Gaussians_wavelet', *args, **kwargs):
        """
        Initialize the DoG wavelet.

        Parameters
        ----------
        sigma_1 : float, optional
            Standard deviation of the first Gaussian. Must be >= 0. Default is 1.0.
        sigma_2 : float, optional
            Standard deviation of the second Gaussian. Must be >= 0. Default is 2.0.
        name : str, optional
            Name of the wavelet. Default is 'difference_of_Gaussians_wavelet'.
        *args, **kwargs
            Additional parameters.
        """
        super().__init__(name=name, *args, **kwargs)
        if sigma_1 < 0.0 or sigma_2 < 0.0:
            raise ValueError('sigma_1 and sigma_2 must be >= 0.')
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

    def psi(self, tau: torch.Tensor):
        """
        Define the DoG wavelet function.

        Parameters
        ----------
        tau : torch.Tensor
            Transformed input values.

        Returns
        -------
        torch.Tensor
            Wavelet values for the input.
        """
        gauss1 = torch.exp(-0.5 * (tau / self.sigma_1) ** 2) / (math.sqrt(2 * torch.pi * self.sigma_1 ** 2))
        gauss2 = torch.exp(-0.5 * (tau / self.sigma_2) ** 2) / (math.sqrt(2 * torch.pi * self.sigma_2 ** 2))
        return gauss1 - gauss2


class meyer_wavelet(discrete_wavelet):
    """
    Meyer wavelet implementation.

    This wavelet has a compactly supported frequency response.
    """
    def __init__(self, name: str = 'meyer_wavelet', *args, **kwargs):
        """
        Initialize the Meyer wavelet.

        Parameters
        ----------
        name : str, optional
            Name of the wavelet. Default is 'meyer_wavelet'.
        *args, **kwargs
            Additional parameters.
        """
        super().__init__(name=name, *args, **kwargs)

    def psi(self, tau: torch.Tensor):
        """
        Define the Meyer wavelet function.

        Parameters
        ----------
        tau : torch.Tensor
            Transformed input values.

        Returns
        -------
        torch.Tensor
            Wavelet values for the input.
        """
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


