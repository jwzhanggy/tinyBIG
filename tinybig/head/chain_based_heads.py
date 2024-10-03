# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################
# Grid Based Head Modules #
###########################

from tinybig.module.base_head import rpn_head
from tinybig.koala.topology.chain import chain
from tinybig.interdependence.topological_interdependence import chain_interdependence
from tinybig.reconciliation.lowrank_reconciliation import lorr_reconciliation
from tinybig.expansion.basic_expansion import identity_expansion
from tinybig.reconciliation.basic_reconciliation import identity_reconciliation
from tinybig.remainder.basic_remainder import zero_remainder, linear_remainder


class recurrent_head(rpn_head):
    def __init__(
        self,
        m: int, n: int,
        chain_length: int,
        name: str = 'recurrent_head',
        bi_directional: bool = False,
        normalization: bool = False,
        normalization_mode: str = 'row_column',
        self_dependence: bool = False,
        require_data: bool = False,
        require_parameters: bool = False,
        channel_num: int = 1,
        with_lorr: bool = False, r: int = 3,
        with_residual: bool = True,
        enable_bias: bool = False,
        device: str = 'cpu',
        *args, **kwargs
    ):
        chain_structure = chain(
            length=chain_length,
            name=name,
            directed=not bi_directional,
            device=device,
        )
        instance_interdependence = chain_interdependence(
            chain=chain_structure,
            bi_directional=bi_directional,
            normalization=normalization,
            normalization_mode=normalization_mode,
            self_dependence=self_dependence,
            require_data=require_data,
            require_parameters=require_parameters,
        )
        data_transformation = identity_expansion(
            device=device
        )

        if with_lorr:
            parameter_fabrication = lorr_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device
            )
        else:
            parameter_fabrication = identity_reconciliation(
                enable_bias=enable_bias,
                device=device
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
            batch_num=chain_length,
            instance_interdependence=instance_interdependence,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            channel_num=channel_num,
            device=device, *args, **kwargs
        )