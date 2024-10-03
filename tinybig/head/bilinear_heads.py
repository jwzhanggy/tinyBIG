# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###################################
# Bilinear Attention Head Modules #
###################################

import torch

from tinybig.module.base_head import rpn_head
from tinybig.expansion.basic_expansion import identity_expansion
from tinybig.reconciliation.basic_reconciliation import identity_reconciliation
from tinybig.remainder.basic_remainder import linear_remainder
from tinybig.interdependence.parameterized_bilinear_interdependence import lowrank_parameterized_bilinear_interdependence


class attention_head(rpn_head):

    def __init__(
        self, m: int, n: int,
        name: str = 'perceptron_head',
        enable_bias: bool = False,
        # optional parameters
        r: int = 3,
        channel_num: int = 1,
        with_batch_norm: bool = False,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        instance_interdependence = lowrank_parameterized_bilinear_interdependence(
            r=r,
            require_data=True,
            require_parameters=True,
            postprocess_functions=[torch.nn.Softmax(dim=0)],
            device=device,
        )

        data_transformation = identity_expansion(
            device=device,
        )

        parameter_fabrication = identity_reconciliation(
            enable_bias=enable_bias,
            device=device,
        )

        remainder = linear_remainder(
            device=device
        )

        output_process_functions = [torch.nn.BatchNorm1d(num_features=n, device=device)] if with_batch_norm else None

        super().__init__(
            m=m, n=n, name=name,
            instance_interdependence=instance_interdependence,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            channel_num=channel_num,
            device=device, *args, **kwargs
        )


