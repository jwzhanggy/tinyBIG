# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###################################
# Bilinear Attention Head Modules #
###################################

"""
Bilinear RPN based heads.

This module contains the bilinear rpn based heads, including
    bilinear_interdependence_head
"""

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
    """
    A bilinear interdependence-based head for multi-channel modules.

    This head implements bilinear interdependence functions, including optional configurations for dual LPHM, LORR,
    data transformations, parameter reconciliation, and various output processing techniques.

    Attributes
    ----------
    m : int
        Input dimension of the head.
    n : int
        Output dimension of the head.
    batch_num : int, optional
        Batch size for instance interdependence.
    channel_num : int
        Number of channels for multi-channel processing.
    parameters_init_method : str
        Initialization method for parameters.
    device : str
        Device to host the head (e.g., 'cpu' or 'cuda').
    """
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
        """
        Initialize a bilinear interdependence head.

        This constructor allows for fine-grained control over instance interdependence, data transformation, parameter
        reconciliation, remainder function, and output processing configurations.

        Parameters
        ----------
        m : int
            Input dimension of the head.
        n : int
            Output dimension of the head.
        name : str, optional
            Name of the head, default is 'bilinear_interdependence_head'.
        batch_num : int, optional
            Batch size for instance interdependence, default is None.
        channel_num : int, optional
            Number of channels for multi-channel processing, default is 1.
        with_dual_lphm_interdependence : bool, optional
            Whether to use dual LPHM parameterized bilinear interdependence, default is False.
        with_lorr_interdependence : bool, optional
            Whether to use LORR parameterized bilinear interdependence, default is False.
        r_interdependence : int, optional
            Rank for the interdependence function, default is 3.
        with_taylor : bool, optional
            Whether to use Taylor expansion for data transformation, default is False.
        d : int, optional
            Degree of Taylor expansion, default is 2.
        with_dual_lphm : bool, optional
            Whether to use dual LPHM for parameter reconciliation, default is False.
        with_lorr : bool, optional
            Whether to use LORR for parameter reconciliation, default is False.
        r : int, optional
            Rank for parameter reconciliation, default is 3.
        enable_bias : bool, optional
            Whether to enable bias in parameter reconciliation, default is False.
        with_residual : bool, optional
            Whether to include a residual connection in the remainder function, default is False.
        with_batch_norm : bool, optional
            Whether to include batch normalization in output processing, default is False.
        with_relu : bool, optional
            Whether to include ReLU activation in output processing, default is True.
        with_softmax : bool, optional
            Whether to include softmax activation in output processing, default is True.
        with_dropout : bool, optional
            Whether to include dropout in output processing, default is False.
        p : float, optional
            Dropout probability, default is 0.25.
        parameters_init_method : str, optional
            Initialization method for parameters, default is 'xavier_normal'.
        device : str, optional
            Device to host the head, default is 'cpu'.

        Returns
        -------
        None
        """
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


