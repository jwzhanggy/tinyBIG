# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch.nn

from tinybig.expansion import expansion
from tinybig.util.util import get_obj_from_str

#####################
# Nested Expansions #
#####################


class extended_expansion(expansion):
    def __init__(self, name='extended_expansion', expansion_pipeline: list = None, expansion_configs: list = None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.expansion_pipeline = []
        if expansion_pipeline is not None and expansion_pipeline != []:
            self.expansion_pipeline = expansion_pipeline
        elif expansion_configs is not None:
            for expansion_config in expansion_configs:
                expansion_class = expansion_config['expansion_class']
                expansion_parameters = expansion_config['expansion_parameters']
                self.expansion_pipeline.append(get_obj_from_str(expansion_class)(**expansion_parameters))

    def calculate_D(self, m: int):
        D = 0
        for expansion_func in self.expansion_pipeline:
            D += expansion_func.calculate_D(m=m)
        return D

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        x = self.pre_process(x=x)
        expansion = []
        for expansion_func in self.expansion_pipeline:
            expansion.append(expansion_func(x=x, device=device))
        return torch.cat(expansion, dim=1)