# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod
import torch

##############################
# Parameter Fabrication Base #
##############################


class base_fabrication(torch.nn.Module):
    def __init__(self, name='base_fabrication', require_parameters=True, enable_bias=False, device='cpu', *args, **kwargs):
        super().__init__()
        self.name = name
        self.device = device
        self.require_parameters = require_parameters
        self.enable_bias = enable_bias

    def get_name(self):
        return self.name

    @abstractmethod
    def calculate_l(self, n: int, D: int):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass