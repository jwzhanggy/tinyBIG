# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################
# Wavelet based Expansion Functions #
#####################################

import itertools
import numpy as np

import torch.nn

from tinybig.expansion import transformation
from tinybig.koala.fourier import (
    discrete_wavelet,
    harr_wavelet,
    dog_wavelet,
    beta_wavelet,
    ricker_wavelet,
    shannon_wavelet,
    meyer_wavelet
)


class discrete_wavelet_expansion(transformation):

    def __init__(self, name: str = 'discrete_wavelet_expansion', d: int = 1, s: int = 1, t: int = 1, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d
        self.s = s
        self.t = t
        self.wavelet = None

    def calculate_D(self, m: int):
        return np.sum([(m * self.s * self.t) ** d for d in range(1, self.d + 1)])

    def wavelet_x(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        assert self.wavelet is not None and isinstance(self.wavelet, discrete_wavelet)

        combinations = list(itertools.product(range(self.s), range(self.t)))
        combination_index = {comb: idx for idx, comb in enumerate(combinations)}

        expansion = torch.ones(size=[x.size(0), x.size(1), self.s * self.t]).to(device)
        for s, t in combination_index:
            n = combination_index[(s, t)]
            expansion[:, :, n] = self.wavelet(x=x, s=s, t=t)
        expansion = expansion[:, :, :].contiguous().view(x.size(0), -1)
        return expansion

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        wavelet_x = self.wavelet_x(x, device=device, *args, **kwargs)

        if self.d > 1:
            wavelet_x_powers = torch.ones(size=[wavelet_x.size(0), 1]).to(device)
            expansion = torch.Tensor([]).to(device)

            for i in range(1, self.d + 1):
                wavelet_x_powers = torch.einsum('ba,bc->bac', wavelet_x_powers.clone(), wavelet_x).view(wavelet_x_powers.size(0), wavelet_x_powers.size(1) * wavelet_x.size(1))
                expansion = torch.cat((expansion, wavelet_x_powers), dim=1)
        else:
            expansion = wavelet_x

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class harr_wavelet_expansion(discrete_wavelet_expansion):
    def __init__(self, name: str = 'harr_wavelet_expansion', a: float = 1.0, b: float = 1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = harr_wavelet(a=a, b=b)


class beta_wavelet_expansion(discrete_wavelet_expansion):
    def __init__(self, name: str = 'beta_wavelet_expansion', a: float = 1.0, b: float = 1.0, alpha: float = 1.0, beta: float = 1.0,  *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = beta_wavelet(a=a, b=b, alpha=alpha, beta=beta)


class shannon_wavelet_expansion(discrete_wavelet_expansion):
    def __init__(self, name: str = 'shannon_wavelet_expansion', a: float = 1.0, b: float = 1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = shannon_wavelet(a=a, b=b)


class ricker_wavelet_expansion(discrete_wavelet_expansion):
    def __init__(self, name: str = 'ricker_wavelet_expansion', a: float = 1.0, b: float = 1.0, sigma: float = 1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = ricker_wavelet(a=a, b=b, sigma=sigma)


class dog_wavelet_expansion(discrete_wavelet_expansion):
    def __init__(self, name: str = 'dog_wavelet_expansion', a: float = 1.0, b: float = 1.0, sigma_1: float = 1.0, sigma_2: float = 2.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = dog_wavelet(a=a, b=b, sigma_1=sigma_1, sigma_2=sigma_2)


class meyer_wavelet_expansion(discrete_wavelet_expansion):
    def __init__(self, name: str = 'meyer_wavelet_expansion', a: float = 1.0, b: float = 1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = meyer_wavelet(a=a, b=b)
