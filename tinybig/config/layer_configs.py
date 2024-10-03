# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################
# RPN Layer Configs #
#####################

from tinybig.config.base_config import config
from tinybig.config.head_configs import perceptron_head_configs


class layer_configs(config):
    def __init__(
        self,
        name: str = "rpn_layer",
        configs: dict = None,
        parameters: dict = None,
        m: int = None, n: int = None,
        head_configs: dict | list = None,
        width: int = None,
        width_alloc: int | list = None,
        head_fusion_configs: dict = None,
        device='cpu',
        *args, **kwargs
    ):
        if configs is not None:
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        elif parameters is not None:
            configs = {
                'layer_class': 'tinybig.module.rpn_head',
                'layer_parameters': parameters,
            }
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        elif m is not None and n is not None and head_configs is not None:
            parameters = {
                'name': name,
                'm': m,
                'n': n,
                'width': width,
                'width_alloc': width_alloc,
                'head_configs': head_configs,
                'head_fusion_configs': head_fusion_configs,
                'device': device,
            }
            configs = {
                'layer_class': 'tinybig.module.rpn_head',
                'layer_parameters': parameters,
            }
            super().__init__(name=name, configs=configs, device=device, *args, **kwargs)
        else:
            super().__init__(name=name, configs=None, device=device, *args, **kwargs)

    def to_instantiation(self):
        if self.configs is None:
            raise ValueError('The layer configs object cannot be none...')
        self.process_width_alloc_head_configs()

        return config.instantiation_from_configs(
            configs=self.configs,
            device=self.device,
            class_name='layer_class',
            parameter_name='layer_parameters'
        )

    def process_width_alloc_head_configs(self):
        assert self.configs is not None

        width = self.configs['width'] if 'width' in self.configs else None
        width_alloc = self.configs['width_alloc'] if 'width_alloc' in self.configs else None
        head_configs = self.configs['head_configs'] if 'head_configs' in self.configs else None

        width, width_alloc, head_configs = config.process_num_alloc_configs(
            num=width,
            num_alloc=width_alloc,
            configs=head_configs,
        )

        self.configs['width'] = width
        self.configs['width_alloc'] = width_alloc
        self.configs['head_configs'] = head_configs


class perceptron_layer_configs(layer_configs):
    def __init__(
        self,
        m: int, n: int,
        name: str = 'perceptron_head_configs',
        enable_bias: bool = False,
        device: str = 'cpu',
        # optional parameters
        with_taylor: bool = False,
        d: int = 2,
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        *args, **kwargs
    ):
        head_configs = [
            perceptron_head_configs(
                m=m, n=n,
                enable_bias=enable_bias,
                device=device,
                with_taylor=with_taylor,
                with_lorr=with_lorr,
                with_residual=with_residual,
                d=d, r=r,
                *args, **kwargs
            ).get_configs()
        ]
        super().__init__(name=name, head_configs=head_configs, device=device, *args, **kwargs)
