# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch.nn

from tinybig.module.transformation import base_transformation as base_expansion

###################################################
# Expansions defined with closed-form polynomials #
###################################################


class hyperbolic_expansion(base_expansion):
    def __init__(self, name='hyperbolic_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 3

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        sinh = torch.sinh(x)
        cosh = torch.cosh(x)
        tanh = torch.tanh(x)
        expansion = torch.cat((sinh, cosh, tanh), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class arc_hyperbolic_expansion(base_expansion):
    def __init__(self, name='hyperbolic_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 3

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        # pre-normalize the input to range [0, 1]
        x = torch.nn.functional.sigmoid(x)
        arcsinh = torch.arcsinh(x)
        arccosh = torch.arccosh(x+1.01)
        arctanh = torch.arctanh(0.99*x)
        expansion = torch.cat((arcsinh, arccosh, arctanh), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class trigonometric_expansion(base_expansion):
    def __init__(self, name='trigonometric_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 3

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        sin = torch.sin(x)
        cos = torch.cos(x)
        tan = torch.tan(x)
        expansion = torch.cat((sin, cos, tan), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)


class arc_trigonometric_expansion(base_expansion):
    def __init__(self, name='trigonometric_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 3

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        # pre-normalize the input to range [0, 1]
        x = torch.nn.functional.sigmoid(x)
        arcsin = torch.arcsin(0.99*x)
        arccos = torch.arccos(0.99*x)
        arctan = torch.arctan(x)
        expansion = torch.cat((arcsin, arccos, arctan), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device)