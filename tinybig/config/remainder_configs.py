# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##############################
# Remainder Function Configs #
##############################

from tinybig.config.function_configs import function_configs


class zero_remainder_configs(function_configs):
    def __init__(self, device: str = 'cpu', *args, **kwargs):
        configs = {
            'function_class': 'tinybig.remainder.zero_remainder',
            'function_parameters': {
                'name': 'zero_remainder',
                'device': device,
            }
        }
        super().__init__(configs=configs, device=device, *args, **kwargs)


class linear_remainder_configs(function_configs):
    def __init__(self, enable_bias: bool = False, device: str = 'cpu', *args, **kwargs):
        configs = {
            'function_class': 'tinybig.remainder.linear_remainder',
            'function_parameters': {
                'name': 'zero_remainder',
                'enable_bias': enable_bias,
                'device': device,
            }
        }
        super().__init__(configs=configs, device=device, *args, **kwargs)