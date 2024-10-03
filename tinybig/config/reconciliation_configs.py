# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############################################
# Parameter Reconciliation Function Configs #
#############################################

from tinybig.config.function_configs import function_configs


class identity_reconciliation_configs(function_configs):
    def __init__(self, enable_bias: bool = False, device: str = 'cpu', *args, **kwargs):
        configs = {
            'function_class': 'tinybig.reconciliation.identity_reconciliation',
            'function_parameters': {
                'name': 'identity_reconciliation',
                'enable_bias': enable_bias,
                'device': device,
            }
        }
        super().__init__(configs=configs, device=device, *args, **kwargs)


class lorr_reconciliation_configs(function_configs):
    def __init__(self, r: int = 3, enable_bias: bool = False, device: str = 'cpu', *args, **kwargs):
        configs = {
            'function_class': 'tinybig.reconciliation.lorr_reconciliation',
            'function_parameters': {
                'name': 'lorr_reconciliation',
                'enable_bias': enable_bias,
                'r': r,
                'device': device,
            }
        }
        super().__init__(configs=configs, device=device, *args, **kwargs)

