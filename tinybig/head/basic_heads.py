# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# RPN Multi-Channel Head Modules #
##################################

import torch

from tinybig.module.base_head import rpn_head
from tinybig.expansion import (
    identity_expansion,
    taylor_expansion,
    bspline_expansion,
    gaussian_rbf_expansion,
    inverse_quadratic_rbf_expansion,
    naive_gamma_expansion,
    naive_cauchy_expansion,
    naive_normal_expansion,
    naive_laplace_expansion,
    naive_exponential_expansion,
    naive_chi2_expansion,
    combinatorial_normal_expansion,
)

from tinybig.reconciliation import identity_reconciliation, lorr_reconciliation, dual_lphm_reconciliation
from tinybig.remainder import zero_remainder, linear_remainder


class perceptron_head(rpn_head):

    def __init__(
        self, m: int, n: int,
        name: str = 'perceptron_head',
        channel_num: int = 1,
        # data transformation function parameters
        with_taylor: bool = False, d: int = 2,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = True,
        # remainder function parameters
        with_residual: bool = False,
        # output processing function parameters
        with_batch_norm: bool = False,
        with_relu: bool = False,
        with_dropout: bool = True, p: float = 0.5,
        with_softmax: bool = False,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if with_taylor:
            data_transformation = taylor_expansion(
                d=d,
                device=device,
            )
        else:
            data_transformation = identity_expansion(
                device=device,
            )

        if with_dual_lphm:
            parameter_fabrication = dual_lphm_reconciliation(
                r=r,
                device=device
            )
        elif with_lorr:
            parameter_fabrication = lorr_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device,
            )
        else:
            parameter_fabrication = identity_reconciliation(
                enable_bias=enable_bias,
                device=device,
            )

        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )

        output_process_functions = []
        if with_batch_norm:
            output_process_functions.append(torch.nn.BatchNorm1d(num_features=n, device=device))
        if with_relu:
            output_process_functions.append(torch.nn.ReLU())
        if with_dropout:
            output_process_functions.append(torch.nn.Dropout(p=p))
        if with_softmax:
            output_process_functions.append(torch.nn.Softmax(dim=-1))

        print('perceptron layer', output_process_functions)

        super().__init__(
            m=m, n=n, name=name,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            channel_num=channel_num,
            device=device, *args, **kwargs
        )


class svm_head(rpn_head):

    def __init__(
        self, m: int, n: int,
        name: str = 'svm_head',
        kernel: str = 'linear',
        base_range: tuple = (-1, 1),
        num_interval: int = 10,
        epsilon: float = 1.0,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        with_batch_norm: bool = False,
        with_softmax: bool = False,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if kernel == 'linear':
            data_transformation = identity_expansion(
                device=device,
            )
        elif kernel == 'gaussian_rbf':
            data_transformation = gaussian_rbf_expansion(
                base_range=base_range,
                num_interval=num_interval,
                epsilon=epsilon,
                device=device,
            )
        elif kernel == 'inverse_quadratic_rbf':
            data_transformation = inverse_quadratic_rbf_expansion(
                base_range=base_range,
                num_interval=num_interval,
                epsilon=epsilon,
                device=device,
            )
        else:
            raise ValueError('kernel must be linear or gaussian_rbf or inverse_quadratic_rbf...')

        if with_lorr:
            parameter_fabrication = lorr_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device,
            )
        else:
            parameter_fabrication = identity_reconciliation(
                enable_bias=enable_bias,
                device=device,
            )

        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )

        output_process_functions = []
        if with_batch_norm:
            output_process_functions.append(torch.nn.BatchNorm1d(num_features=n, device=device))
        if with_softmax:
            output_process_functions.append(torch.nn.Softmax(dim=-1))

        super().__init__(
            m=m, n=n, name=name,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            channel_num=channel_num,
            device=device, *args, **kwargs
        )


class kan_head(rpn_head):

    def __init__(
        self, m: int, n: int,
        grid_range=(-1, 1), t: int = 5, d: int = 3,
        name: str = 'kan_head',
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False, r: int = 3,
        channel_num: int = 1,
        with_batch_norm: bool = False,
        with_softmax: bool = False,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        data_transformation = bspline_expansion(
            grid_range=grid_range,
            t=t, d=d,
            device=device,
        )

        if with_lorr:
            parameter_fabrication = lorr_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device,
            )
        else:
            parameter_fabrication = identity_reconciliation(
                enable_bias=enable_bias,
                device=device,
            )

        remainder = linear_remainder(
            require_remainder_parameters=True,
            activation_functions=[torch.nn.SiLU()],
            device=device,
        )

        output_process_functions = []
        if with_batch_norm:
            output_process_functions.append(torch.nn.BatchNorm1d(num_features=n, device=device))
        if with_softmax:
            output_process_functions.append(torch.nn.Softmax(dim=-1))

        super().__init__(
            m=m, n=n, name=name,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            channel_num=channel_num,
            device=device, *args, **kwargs
        )


class naive_bayes_head(rpn_head):

    def __init__(
        self, m: int, n: int,
        name: str = 'perceptron_head',
        distribution: str = 'normal',
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if distribution == 'normal':
            data_transformation = naive_normal_expansion(
                device=device,
            )
        elif distribution == 'exponential':
            data_transformation = naive_exponential_expansion(
                device=device,
            )
        elif distribution == 'cauchy':
            data_transformation = naive_cauchy_expansion(
                device=device,
            )
        elif distribution == 'gamma':
            data_transformation = naive_gamma_expansion(
                device=device,
            )
        elif distribution == 'chi2':
            data_transformation = naive_chi2_expansion(
                device=device,
            )
        elif distribution == 'laplace':
            data_transformation = naive_laplace_expansion(
                device=device,
            )
        else:
            raise ValueError('tinybig only supports normal, exponential, cauchy, gamma, laplace or chi2 distributions...')

        if with_lorr:
            parameter_fabrication = lorr_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device,
            )
        else:
            parameter_fabrication = identity_reconciliation(
                enable_bias=enable_bias,
                device=device,
            )

        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )

        super().__init__(
            m=m, n=n, name=name,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            channel_num=channel_num,
            device=device, *args, **kwargs
        )


class pgm_head(rpn_head):

    def __init__(
        self, m: int, n: int,
        name: str = 'perceptron_head',
        distribution: str = 'normal',
        d: int = 2, with_replacement: bool = False,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if distribution == 'normal':
            data_transformation = combinatorial_normal_expansion(
                d=d, with_replacement=with_replacement,
                device=device,
            )
        else:
            raise ValueError('tinybig only supports normal, exponential, cauchy, gamma, laplace or chi2 distributions...')

        if with_lorr:
            parameter_fabrication = lorr_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device,
            )
        else:
            parameter_fabrication = identity_reconciliation(
                enable_bias=enable_bias,
                device=device,
            )

        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )

        super().__init__(
            m=m, n=n, name=name,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            channel_num=channel_num,
            device=device, *args, **kwargs
        )