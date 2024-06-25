# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod
import torch

from tinybig.util import func_x, process_function_list, register_function_parameters

####################
#  Transformation Base  #
####################


class base_transformation(torch.nn.Module):
    def __init__(self, name='base_transformation', input_dimension: int = None, preprocess_functions=None, postprocess_functions=None,
                 preprocess_function_configs=None, postprocess_function_configs=None, device='cpu', *args, **kwargs):
        super().__init__()
        self.name = name
        self.input_dimension = input_dimension
        self.device = device
        self.preprocess_functions = process_function_list(preprocess_functions, preprocess_function_configs, device=self.device)
        self.postprocess_functions = process_function_list(postprocess_functions, postprocess_function_configs, device=self.device)
        #register_function_parameters(self, self.preprocess_functions)
        #register_function_parameters(self, self.postprocess_functions)

    def get_name(self):
        return self.name

    def pre_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return func_x(x, self.preprocess_functions, device=device)

    def post_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return func_x(x, self.postprocess_functions, device=device)

    @abstractmethod
    def calculate_D(self, m: int):
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        pass