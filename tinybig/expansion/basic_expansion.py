# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch.nn

from tinybig.module.transformation import base_transformation as base_expansion


####################
# Basic Expansions #
####################


class identity_expansion(base_expansion):
    def __init__(self, name='identity_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        expansion = x
        return self.post_process(x=expansion, device=device)


class reciprocal_expansion(base_expansion):
    def __init__(self, name='reciprocal_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x[torch.logical_and(x>=0, x<=1e-6)] = 1e-6
        x[torch.logical_and(x<0, x>=-1e-6)] = -1e-6
        expansion = torch.reciprocal(x)
        return self.post_process(x=expansion, device=device)


class linear_expansion(base_expansion):
    def __init__(self, name='linear_expansion', c=None, pre_C=None, post_C=None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.c = c
        self.pre_C = pre_C
        self.post_C = post_C

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', c=None, pre_C=None, post_C=None, *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        c = c if c is not None else self.c
        pre_C = pre_C if pre_C is not None else self.pre_C
        post_C = post_C if post_C is not None else self.post_C

        if c is not None:
            expansion = c * x
        elif pre_C is not None:
            assert pre_C.size(-1) == x.size(0)
            expansion = torch.matmul(pre_C, x)
        elif post_C is not None:
            assert x.size(-1) == post_C.size(0)
            expansion = torch.matmul(x, post_C)
        else:
            expansion = x

        return self.post_process(x=expansion, device=device)


if __name__== '__main__':
    import torch.nn.functional as F
    exp = reciprocal_expansion(postprocess_functions=F.sigmoid)
    x = torch.Tensor([[0.5, 0.5]])
    print(x, exp(x))