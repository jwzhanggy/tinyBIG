# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###################################
# Bilinear Attention Head Modules #
###################################

import torch
import math
from functools import partial

from tinybig.module.base_head import head
from tinybig.koala.linear_algebra import (
    operator_based_normalize_matrix
)
from tinybig.koala.algebra import (
    find_close_factors
)

from tinybig.expansion import (
    identity_expansion,
    taylor_expansion
)
from tinybig.reconciliation import (
    identity_reconciliation,
    lorr_reconciliation,
    dual_lphm_reconciliation
)
from tinybig.remainder import (
    linear_remainder,
    zero_remainder
)
from tinybig.interdependence import (
    parameterized_bilinear_interdependence,
    lowrank_parameterized_bilinear_interdependence,
    lphm_parameterized_bilinear_interdependence,
    hm_parameterized_bilinear_interdependence,
    dual_lphm_parameterized_bilinear_interdependence,
    random_matrix_adaption_parameterized_bilinear_interdependence
)


class bilinear_interdependence_head(head):

    def __init__(
        self,
        m: int, n: int,
        name: str = 'bilinear_interdependence_head',
        batch_num: int = None,
        channel_num: int = 1,
        # interdependence function parameters
        with_dual_lphm_interdependence: bool = False,
        with_lorr_interdependence: bool = False, r_interdependence: int = 3,
        # data transformation function parameters
        with_taylor: bool = False, d: int = 2,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = False,
        # remainder function parameters
        with_residual: bool = False,
        # output processing parameters
        with_batch_norm: bool = False,
        with_relu: bool = True,
        with_softmax: bool = True,
        with_dropout: bool = False, p: float = 0.25,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):

        # instance interdependence function
        if with_lorr_interdependence:
            instance_interdependence = lowrank_parameterized_bilinear_interdependence(
                b=batch_num, m=m,
                r=r_interdependence,
                interdependence_type='instance',
                require_data=True,
                require_parameters=True,
                postprocess_functions=[
                    partial(
                        operator_based_normalize_matrix,
                        mask_zero=True,
                        operator=torch.nn.functional.softmax,
                        rescale_factor=math.sqrt(n),
                        mode='column'
                    )
                ],
                device=device,
            )
        elif with_dual_lphm_interdependence:
            instance_interdependence = dual_lphm_parameterized_bilinear_interdependence(
                b=batch_num, m=m,
                p=find_close_factors(m), r=r_interdependence,
                interdependence_type='instance',
                require_data=True,
                require_parameters=True,
                postprocess_functions=[
                    partial(
                        operator_based_normalize_matrix,
                        mask_zero=True,
                        operator=torch.nn.functional.softmax,
                        rescale_factor=math.sqrt(n),
                        mode='column'
                    )
                ],
                device=device,
            )
        else:
            instance_interdependence = parameterized_bilinear_interdependence(
                b=batch_num, m=m,
                interdependence_type='instance',
                require_data=True,
                require_parameters=True,
                postprocess_functions=[
                    partial(
                        operator_based_normalize_matrix,
                        mask_zero=True,
                        operator=torch.nn.functional.softmax,
                        rescale_factor=math.sqrt(n),
                        mode='column'
                    )
                ],
                device=device,
            )

        # data transformation function
        if with_taylor:
            data_transformation = taylor_expansion(
                d=d,
                device=device,
            )
        else:
            data_transformation = identity_expansion(
                device=device,
            )

        # parameter reconciliation function
        if with_dual_lphm:
            print('bilinear head', 'with_dual_lphm:', with_dual_lphm, 'r:', r)
            parameter_fabrication = dual_lphm_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device
            )
        elif with_lorr:
            print('bilinear head', 'with_lorr:', with_dual_lphm, 'r:', r)
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

        # remainder function
        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )

        # output processing function
        output_process_functions = []
        if with_batch_norm:
            output_process_functions.append(torch.nn.BatchNorm1d(num_features=n, device=device))
        if with_relu:
            output_process_functions.append(torch.nn.ReLU())
        if with_dropout:
            output_process_functions.append(torch.nn.Dropout(p=p))
        if with_softmax:
            output_process_functions.append(torch.nn.Softmax(dim=-1))

        super().__init__(
            m=m, n=n, name=name,
            instance_interdependence=instance_interdependence,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            channel_num=channel_num,
            parameters_init_method=parameters_init_method,
            device=device, *args, **kwargs
        )


