# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch
from scipy.special import comb

from tinybig.module.transformation import base_transformation as base_expansion


############################
# Combinatorial Expansions #
############################

class combinatorial_expansion(base_expansion):
    def __init__(self, name='combinatorial_expansion', d: int = 1, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.d = d

    def calculate_D(self, m: int):
        assert type(self.d) is int and self.d >= 1
        return int(sum([comb(m, r) for r in range(1, self.d+1)]))

    @staticmethod
    def random_combinations(x: torch.Tensor, r: int = 2, with_replacement: bool = False):
        assert 0 <= r <= x.numel()
        return torch.combinations(input=x, r=r, with_replacement=with_replacement)

    @staticmethod
    def combinatorial(x: torch.Tensor, d=2, device='cpu', *args, **kwargs):
        # x: [batch, m]
        if len(x.shape) == 2:
            expansion = []
            for r in range(1, d + 1):
                degree_expansion = []
                for i in range(x.size(0)):
                    degree_expansion.append(combinatorial_expansion.random_combinations(x=x[i,:], r=r))
                expansion.append(degree_expansion)
            # expansion: [0:d-1, 0:batch-1, 0:(m choose d)-1]
            return expansion
        elif len(x.shape) == 1:
            # x: [m]
            expansion = []
            for r in range(1, d + 1):
                expansion.append(combinatorial_expansion.random_combinations(x=x, r=r))
            # expansion: [0:d-1, 0:(m choose d)-1]
            return expansion
        else:
            raise ValueError("Input x can only be 2d or 1d, higher dimensional inputs are not supported yet...")

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return self.combinatorial(x=x, d=self.d, device=device, *args, **kwargs)


class combinatorial_normal_expansion(combinatorial_expansion):
    def __init__(self, name='combinatorial_normal_expansion', d: int = 1, *args, **kwargs):
        print('combinatorial_normal_expansion initialization')
        super().__init__(name=name, d=d, *args, **kwargs)

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        expansion_shape = list(x.size())
        expansion_shape[-1] = self.calculate_D(m=expansion_shape[-1])

        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)
        if len(x.shape) >= 3:
            raise ValueError("Input x can only be 2d or 1d, higher dimensional inputs are not supported yet...")

        x = x.to('cpu')
        combinations = self.combinatorial(x=x, d=self.d, device=device, *args, **kwargs)
        result = torch.zeros(x.size(0), self.calculate_D(m=x.size(1)))
        distribution_dict = {}
        current_index = 0
        for r in range(1, self.d+1):
            if r not in distribution_dict:
                # multivariate_normal_distributions
                distribution_dict[r] = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=torch.zeros(r), covariance_matrix=torch.eye(r)
                )
            degree_batch_expansion = torch.stack(combinations[r-1], dim=0)
            tuple_count = len(degree_batch_expansion[0])
            degree_batch_log_likelihood = distribution_dict[r].log_prob(value=degree_batch_expansion)
            result[:, current_index:current_index+tuple_count] = degree_batch_log_likelihood
            current_index += tuple_count
        return self.post_process(x=result.view(*expansion_shape), device=device).to(device)



if __name__ == "__main__":
    import time
    start_time = time.time()
    a = torch.Tensor([1, 2, 3, 4])
    b = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    rcf = combinatorial_normal_expansion(d=4)#, postprocess_functions='layer_norm', postprocess_functions_parameters={'dim': 1})
    print(rcf(a, device='mps'), time.time()-start_time)
