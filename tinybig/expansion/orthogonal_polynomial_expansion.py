# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###################################################
# Orthogonal Polynomial based Expansion Functions #
###################################################

import torch.nn

from tinybig.expansion import transformation


class hermite_expansion(transformation):

    def __init__(self, name: str = 'chebyshev_polynomial_expansion', d: int = 2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = x * expansion[:, :, n-1].clone() - (n-1) * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class laguerre_expansion(transformation):

    def __init__(self, name='laguerre_polynomial_expansion', d: int = 2, alpha: float = 1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d
        self.alpha = alpha

    def calculate_D(self, m: int):
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x + 1.0 + self.alpha
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (2*n-1+self.alpha-x)/n * expansion[:, :, n-1].clone() - (n-1+self.alpha)/n * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class legendre_expansion(transformation):

    def __init__(self, name='legendre_polynomial_expansion', d: int = 2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (2*n-1)/n * x * expansion[:, :, n-1].clone() - (n-1)/n * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class gegenbauer_expansion(transformation):

    def __init__(self, name='gegenbauer_polynomial_expansion', d: int = 2, alpha: float = 1.0, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d
        self.alpha = alpha

    def calculate_D(self, m: int):
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = 2 * self.alpha * x
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (n-1+self.alpha)/n * 2*x * expansion[:, :, n-1].clone() - (n-2+2*self.alpha)/n * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class bessel_expansion(transformation):

    def __init__(self, name='bessel_polynomial_expansion', d: int = 2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x + 1
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (2*n-1) * x * expansion[:, :, n-1].clone() + expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class reverse_bessel_expansion(transformation):

    def __init__(self, name='reverse_bessel_polynomial_expansion', d: int = 2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x + 1
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = (2*n-1) * expansion[:, :, n-1].clone() + x * x * expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class fibonacci_expansion(transformation):

    def __init__(self, name='fibonacci_polynomial_expansion', d: int = 2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = torch.zeros(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = torch.ones(size=[x.size(0), x.size(1)]).to(device)
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = x * expansion[:, :, n-1].clone() + expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class lucas_expansion(transformation):

    def __init__(self, name='lucas_polynomial_expansion', d: int = 2, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        return m * self.d

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        # base case: order 0
        expansion = 2 * torch.ones(size=[x.size(0), x.size(1), self.d + 1]).to(device)
        # base case: order 1
        if self.d > 0:
            expansion[:, :, 1] = x
        # high-order cases
        for n in range(2, self.d + 1):
            expansion[:, :, n] = x * expansion[:, :, n-1].clone() + expansion[:, :, n-2].clone()

        expansion = expansion[:, :, 1:].contiguous().view(x.size(0), -1)

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)