# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################
# Chain Based Head Modules #
###########################

import torch

from tinybig.module import rpn_head
from tinybig.koala.topology import chain
from tinybig.interdependence import (
    chain_interdependence,
    multihop_chain_interdependence,
    inverse_approx_multihop_chain_interdependence,
    exponential_approx_multihop_chain_interdependence
)
from tinybig.reconciliation import (
    identity_reconciliation,
    lorr_reconciliation,
    dual_lphm_reconciliation
)
from tinybig.expansion import (
    identity_expansion,
    taylor_expansion
)
from tinybig.remainder import (
    zero_remainder,
    linear_remainder
)


class recurrent_head(rpn_head):
    def __init__(
        self,
        m: int, n: int,
        chain_length: int,
        channel_num: int = 1,
        name: str = 'recurrent_head',
        # interdependence function parameters
        bi_directional: bool = False,
        with_multihop: bool = False, h: int = 1, accumulative: bool = False,
        with_inverse_approx: bool = False,
        with_exponential_approx: bool = False,
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
        device: str = 'cpu', *args, **kwargs
    ):
        chain_structure = chain(
            length=chain_length,
            name=name,
            bi_directional=bi_directional,
            device=device,
        )

        if with_exponential_approx:
            instance_interdependence = exponential_approx_multihop_chain_interdependence(
                b=chain_length, m=m,
                chain=chain_structure,
                interdependence_type='instance',
                normalization=False,
                self_dependence=True,
                require_data=False,
                require_parameters=False,
            )
        elif with_inverse_approx:
            instance_interdependence = inverse_approx_multihop_chain_interdependence(
                b=chain_length, m=m,
                chain=chain_structure,
                interdependence_type='instance',
                normalization=False,
                self_dependence=True,
                require_data=False,
                require_parameters=False,
            )
        elif with_multihop:
            instance_interdependence = multihop_chain_interdependence(
                h=h, accumulative=accumulative,
                b=chain_length, m=m,
                chain=chain_structure,
                interdependence_type='instance',
                normalization=False,
                self_dependence=True,
                require_data=False,
                require_parameters=False,
            )
        else:
            instance_interdependence = chain_interdependence(
                b=chain_length, m=m,
                chain=chain_structure,
                interdependence_type='instance',
                normalization=False,
                self_dependence=True,
                require_data=False,
                require_parameters=False,
            )

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
                enable_bias=enable_bias,
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

        print('recurrent head', output_process_functions)

        super().__init__(
            m=m, n=n, name=name, channel_num=channel_num,
            batch_num=chain_length,
            instance_interdependence=instance_interdependence,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            device=device, *args, **kwargs
        )