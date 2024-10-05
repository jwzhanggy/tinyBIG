# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################
# RPN Head Configs #
####################

from tinybig.config.base_config import config

from tinybig.config.transformation_configs import (
    identity_expansion_configs,
    taylor_expansion_configs,
)

from tinybig.config.reconciliation_configs import (
    identity_reconciliation_configs,
    lorr_reconciliation_configs,
)

from tinybig.config.remainder_configs import (
    zero_remainder_configs,
    linear_remainder_configs,
)


class head_configs(config):
    def __init__(
        self,
        name: str = 'rpn_head',
        configs: dict = None,
        parameters: dict = None,
        m: int = None, n: int = None,
        batch: int = None,
        l: int = None,
        channel_num: int = 1,

        data_transformation_configs: dict = None,
        parameter_fabrication_configs: dict = None,
        remainder_configs: dict = None,

        channel_fusion_configs: dict = None,

        attribute_interdependence_configs: dict = None,
        instance_interdependence_configs: dict = None,

        input_processing_configs: dict | list = None,
        output_processing_configs: dict | list = None,

        device='cpu',
        *args, **kwargs
    ):
        if configs is not None:
            assert isinstance(configs, dict) and 'head_class' in configs and configs['head_class'] == 'tinybig.module.rpn_head'
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        elif parameters is not None:
            configs = {
                'head_class': 'tinybig.module.rpn_head',
                'head_parameters': parameters,
            }
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        elif m is not None and n is not None:
            parameters = {
                'name': name,
                'm': m,
                'n': n,
                'l': l,
                'batch': batch,
                'channel_num': channel_num,

                'data_transformation_configs': data_transformation_configs,
                'parameter_fabrication_configs': parameter_fabrication_configs,
                'remainder_configs': remainder_configs,
                'channel_fusion_configs': channel_fusion_configs,
                'attribute_interdependence_configs': attribute_interdependence_configs,
                'instance_interdependence_configs': instance_interdependence_configs,
                'input_processing_configs': input_processing_configs,
                'output_processing_configs': output_processing_configs,
                'device': device,
            }
            configs = {
                'head_class': 'tinybig.module.rpn_head',
                'head_parameters': parameters,
            }
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        else:
            super().__init__(name=name, configs=None, device=device, *args, **kwargs)

    def to_instantiation(self):
        if self.configs is None:
            raise ValueError('The head configs object cannot be none...')

        return config.instantiation_from_configs(
            configs=self.configs,
            device=self.device,
            class_name='head_class',
            parameter_name='head_parameters'
        )


class perceptron_head_configs(head_configs):
    def __init__(
        self,
        m: int, n: int,
        name: str = 'perceptron_head_configs',
        enable_bias: bool = False,
        device='cpu',
        # optional parameters
        with_taylor: bool = False,
        d: int = 2,
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        *args, **kwargs
    ):
        if with_taylor:
            data_transformation_configs = taylor_expansion_configs(
                d=d,
                device=device
            )
        else:
            data_transformation_configs = identity_expansion_configs(device=device)

        if with_lorr:
            parameter_fabrication_configs = lorr_reconciliation_configs(
                r=r,
                enable_bias=enable_bias,
                device=device
            )
        else:
            parameter_fabrication_configs = identity_reconciliation_configs(
                    enable_bias=enable_bias,
                    device=device
            )

        if with_residual:
            remainder_configs = linear_remainder_configs(device=device)
        else:
            remainder_configs = zero_remainder_configs(device=device)

        parameters = {
            'name': name,
            'm': m,
            'n': n,
            'device': device,

            'data_transformation_configs': data_transformation_configs,
            'parameter_fabrication_configs': parameter_fabrication_configs,
            'remainder_configs': remainder_configs,
        }
        super().__init__(class_name='tinybig.module.rpn_head', parameters=parameters, device=device, *args, **kwargs)