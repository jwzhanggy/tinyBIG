# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch
import torch.nn.functional as F

from tinybig.expansion import expansion

############################################################
# Expansions defined with pure probabilistic distributions #
############################################################


class naive_normal_expansion(expansion):
    def __init__(self, name='naive_normal_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        normal_dist_1 = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        normal_x_1 = normal_dist_1.log_prob(x)

        expansion = normal_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class normal_expansion(expansion):
    def __init__(self, name='normal_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        normal_dist_1 = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        normal_x_1 = normal_dist_1.log_prob(x)

        normal_dist_2 = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([1.0]))
        normal_x_2 = normal_dist_2.log_prob(x)

        normal_dist_3 = torch.distributions.normal.Normal(torch.tensor([-1.0]), torch.tensor([1.0]))
        normal_x_3 = normal_dist_3.log_prob(x)

        normal_dist_4 = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
        normal_x_4 = normal_dist_4.log_prob(x)

        normal_dist_5 = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([0.1]))
        normal_x_5 = normal_dist_5.log_prob(x)

        normal_dist_6 = torch.distributions.normal.Normal(torch.tensor([-1.0]), torch.tensor([0.1]))
        normal_x_6 = normal_dist_6.log_prob(x)

        expansion = torch.cat((normal_x_1, normal_x_2, normal_x_3, normal_x_4, normal_x_5, normal_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_cauchy_expansion(expansion):
    def __init__(self, name='naive_cauchy_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        cauchy_dist_1 = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        cauchy_x_1 = cauchy_dist_1.log_prob(x)

        expansion = cauchy_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class cauchy_expansion(expansion):
    def __init__(self, name='cauchy_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        cauchy_dist_1 = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        cauchy_x_1 = cauchy_dist_1.log_prob(x)

        cauchy_dist_2 = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]), torch.tensor([0.5]))
        cauchy_x_2 = cauchy_dist_2.log_prob(x)

        cauchy_dist_3 = torch.distributions.cauchy.Cauchy(torch.tensor([1.0]), torch.tensor([1.0]))
        cauchy_x_3 = cauchy_dist_3.log_prob(x)

        cauchy_dist_4 = torch.distributions.cauchy.Cauchy(torch.tensor([1.0]), torch.tensor([0.5]))
        cauchy_x_4 = cauchy_dist_4.log_prob(x)

        cauchy_dist_5 = torch.distributions.cauchy.Cauchy(torch.tensor([-1.0]), torch.tensor([1.0]))
        cauchy_x_5 = cauchy_dist_5.log_prob(x)

        cauchy_dist_6 = torch.distributions.cauchy.Cauchy(torch.tensor([-1.0]), torch.tensor([0.5]))
        cauchy_x_6 = cauchy_dist_6.log_prob(x)

        expansion = torch.cat((cauchy_x_1, cauchy_x_2, cauchy_x_3, cauchy_x_4, cauchy_x_5, cauchy_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_chi2_expansion(expansion):
    def __init__(self, name='naive_chi2_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')
        x = F.sigmoid(x)+0.001

        chi2_dist_1 = torch.distributions.chi2.Chi2(df=torch.tensor([1.0]))
        chi2_x_1 = chi2_dist_1.log_prob(x)

        expansion = chi2_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class chi2_expansion(expansion):
    def __init__(self, name='chi2_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 2

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')
        x = F.sigmoid(x)+0.001

        chi2_dist_1 = torch.distributions.chi2.Chi2(df=torch.tensor([1.0]))
        chi2_x_1 = chi2_dist_1.log_prob(x)

        chi2_dist_6 = torch.distributions.chi2.Chi2(df=torch.tensor([2.0]))
        chi2_x_6 = chi2_dist_6.log_prob(x)

        expansion = torch.cat((chi2_x_1, chi2_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_gamma_expansion(expansion):
    def __init__(self, name='naive_gamma_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        # pre-normalize the input to range [0, 1]
        x = x.to('cpu')
        x = 0.99 * F.sigmoid(x) + 0.001

        gamma_dist_1 = torch.distributions.gamma.Gamma(torch.tensor([0.5]), torch.tensor([1.0]))
        gamma_x_1 = gamma_dist_1.log_prob(x)

        expansion = gamma_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class gamma_expansion(expansion):
    def __init__(self, name='gamma_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        # pre-normalize the input to range [0, 1]
        x = x.to('cpu')
        x = 0.99 * F.sigmoid(x) + 0.001

        gamma_dist_1 = torch.distributions.gamma.Gamma(torch.tensor([0.5]), torch.tensor([1.0]))
        gamma_x_1 = gamma_dist_1.log_prob(x)

        gamma_dist_2 = torch.distributions.gamma.Gamma(torch.tensor([0.5]), torch.tensor([2.0]))
        gamma_x_2 = gamma_dist_2.log_prob(x)

        gamma_dist_3 = torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        gamma_x_3 = gamma_dist_3.log_prob(x)

        gamma_dist_4 = torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([2.0]))
        gamma_x_4 = gamma_dist_4.log_prob(x)

        gamma_dist_5 = torch.distributions.gamma.Gamma(torch.tensor([2.0]), torch.tensor([1.0]))
        gamma_x_5 = gamma_dist_5.log_prob(x)

        gamma_dist_6 = torch.distributions.gamma.Gamma(torch.tensor([2.0]), torch.tensor([2.0]))
        gamma_x_6 = gamma_dist_6.log_prob(x)

        expansion = torch.cat((gamma_x_1, gamma_x_2, gamma_x_3, gamma_x_4, gamma_x_5, gamma_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_laplace_expansion(expansion):
    def __init__(self, name='naive_laplace_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        laplace_dist_1 = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        laplace_x_1 = laplace_dist_1.log_prob(x)

        expansion = laplace_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class laplace_expansion(expansion):
    def __init__(self, name='laplace_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        laplace_dist_1 = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        laplace_x_1 = laplace_dist_1.log_prob(x)

        laplace_dist_2 = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([2.0]))
        laplace_x_2 = laplace_dist_2.log_prob(x)

        laplace_dist_3 = torch.distributions.laplace.Laplace(torch.tensor([1.0]), torch.tensor([1.0]))
        laplace_x_3 = laplace_dist_3.log_prob(x)

        laplace_dist_4 = torch.distributions.laplace.Laplace(torch.tensor([1.0]), torch.tensor([2.0]))
        laplace_x_4 = laplace_dist_4.log_prob(x)

        laplace_dist_5 = torch.distributions.laplace.Laplace(torch.tensor([-1.0]), torch.tensor([1.0]))
        laplace_x_5 = laplace_dist_5.log_prob(x)

        laplace_dist_6 = torch.distributions.laplace.Laplace(torch.tensor([-1.0]), torch.tensor([2.0]))
        laplace_x_6 = laplace_dist_6.log_prob(x)

        expansion = torch.cat((laplace_x_1, laplace_x_2, laplace_x_3, laplace_x_4, laplace_x_5, laplace_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class naive_exponential_expansion(expansion):
    def __init__(self, name='naive_exponential_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        # pre-normalize the input to range [0, 1]
        x = x.to('cpu')
        x = 0.99 * F.sigmoid(x) + 0.001

        exponential_dist_1 = torch.distributions.exponential.Exponential(torch.tensor([0.5]))
        exponential_x_1 = exponential_dist_1.log_prob(x)

        expansion = exponential_x_1

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class exponential_expansion(expansion):
    def __init__(self, name='exponential_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        # pre-normalize the input to range [0, 1]
        x = x.to('cpu')
        x = 0.99 * F.sigmoid(x) + 0.001

        exponential_dist_1 = torch.distributions.exponential.Exponential(torch.tensor([0.5]))
        exponential_x_1 = exponential_dist_1.log_prob(x)

        exponential_dist_2 = torch.distributions.exponential.Exponential(torch.tensor([0.75]))
        exponential_x_2 = exponential_dist_2.log_prob(x)

        exponential_dist_3 = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
        exponential_x_3 = exponential_dist_3.log_prob(x)

        exponential_dist_4 = torch.distributions.exponential.Exponential(torch.tensor([2.0]))
        exponential_x_4 = exponential_dist_4.log_prob(x)

        exponential_dist_5 = torch.distributions.exponential.Exponential(torch.tensor([3.0]))
        exponential_x_5 = exponential_dist_5.log_prob(x)

        exponential_dist_6 = torch.distributions.exponential.Exponential(torch.tensor([5.0]))
        exponential_x_6 = exponential_dist_6.log_prob(x)

        expansion = torch.cat((exponential_x_1, exponential_x_2, exponential_x_3, exponential_x_4, exponential_x_5, exponential_x_6), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)


class hybrid_probabilistic_expansion(expansion):
    def __init__(self, name='hybrid_probabilistic_expansion', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def calculate_D(self, m: int):
        return m * 6

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x, device=device)
        x = x.to('cpu')

        normal_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        normal_x = normal_dist.log_prob(x)

        cauchy_dist = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        cauchy_x = cauchy_dist.log_prob(x)

        chi_dist = torch.distributions.chi2.Chi2(df=torch.tensor([1.0]))
        chi_x = chi_dist.log_prob(0.99 * F.sigmoid(x) + 0.001)

        gamma_dist = torch.distributions.gamma.Gamma(torch.tensor([0.5]), torch.tensor([1.0]))
        gamma_x = gamma_dist.log_prob(0.99 * F.sigmoid(x) + 0.001)

        laplace_dist = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        laplace_x = laplace_dist.log_prob(x)

        exponential_dist = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
        exponential_x = exponential_dist.log_prob(0.99 * F.sigmoid(x) + 0.001)

        expansion = torch.cat((normal_x, cauchy_x, chi_x, gamma_x, laplace_x, exponential_x), dim=1)

        assert self.calculate_D(m=x.size(1)) == expansion.size(1)
        return self.post_process(x=expansion, device=device).to(device)
