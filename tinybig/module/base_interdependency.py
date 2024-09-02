# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################
# Interdependency Function Base #
#################################

from abc import abstractmethod
import torch

from tinybig.util import process_function_list, func_x


class interdependency(torch.nn.Module):

    def __init__(
            self,
            name='base_interdependency',
            require_parameters=False,
            enable_bias=False,
            processing_functions=None,
            processing_function_configs=None,
            normalization=None,
            device='cpu',
            *args, **kwargs
    ):
        super().__init__()
        self.name = name
        self.require_parameters = require_parameters
        self.enable_bias = enable_bias
        self.processing_functions = process_function_list(processing_functions, processing_function_configs, device=self.device)
        self.normalization = normalization
        self.device = device

    def get_name(self):
        return self.name

    def processing(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return func_x(x, self.processing_functions, device=device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
