# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

########################################
# Data Transformation Function Configs #
########################################

from tinybig.config.function_configs import function_configs


class identity_expansion_configs(function_configs):
    def __init__(self, device: str = 'cpu', *args, **kwargs):
        configs = {
            'function_class': 'tinybig.expansion.identity_expansion',
            'function_parameters': {
                'name': 'identity_expansion',
                'device': device,
            }
        }
        super().__init__(configs=configs, device=device, *args, **kwargs)


class taylor_expansion_configs(function_configs):
    def __init__(self, d: int = 2, device: str = 'cpu', *args, **kwargs):
        configs = {
            'function_class': 'tinybig.expansion.taylor_expansion',
            'function_parameters': {
                'name': 'taylor_expansion',
                'd': d,
                'device': device,
            }
        }
        super().__init__(configs=configs, device=device, *args, **kwargs)