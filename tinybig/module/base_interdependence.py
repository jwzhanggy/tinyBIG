# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################
# Interdependence Function Base #
#################################

from abc import abstractmethod
import torch

from tinybig.util import process_function_list, func_x


class interdependence(torch.nn.Module):

    def __init__(
            self,
            name='base_interdependency',
            require_parameters=False,
            enable_bias=False,
            preprocess_functions=None,
            postprocess_functions=None,
            preprocess_function_configs=None,
            postprocess_function_configs=None,
            device='cpu',
            *args, **kwargs
    ):
        super().__init__()
        self.name = name
        self.require_parameters = require_parameters
        self.enable_bias = enable_bias
        self.preprocess_functions = process_function_list(preprocess_functions, preprocess_function_configs, device=self.device)
        self.postprocess_functions = process_function_list(postprocess_functions, postprocess_function_configs, device=self.device)
        self.device = device

    def get_name(self):
        return self.name

    def pre_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return func_x(x, self.preprocess_functions, device=device)

    def post_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return func_x(x, self.postprocess_functions, device=device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def calculate_o_prime(self, o: int):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
