# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod
import torch

from tinybig.util import process_function_list, func_x, register_function_parameters

##################
# Remainder Base #
##################


class remainder(torch.nn.Module):

    def __init__(self, name='base_remainder', require_parameters=False, enable_bias=False, activation_functions=None,
                 activation_function_configs=None, device='cpu', *args, **kwargs):
        super().__init__()
        self.name = name
        self.device = device
        self.require_parameters = require_parameters
        self.enable_bias = enable_bias
        self.activation_functions = process_function_list(activation_functions, activation_function_configs, device=self.device)
        #register_function_parameters(self, self.activation_functions)

    def get_name(self):
        return self.name

    def activation(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return func_x(x, self.activation_functions, device=device)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass