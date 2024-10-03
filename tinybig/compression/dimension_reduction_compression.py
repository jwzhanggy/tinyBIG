# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################################
# Dimension Reduction based Compression Function #
##################################################

import torch

from tinybig.compression import transformation
from tinybig.koala.machine_learning.dimension_reduction import incremental_dimension_reduction, incremental_PCA, incremental_random_projection
from tinybig.config.base_config import config


class dimension_reduction_compression(transformation):
    def __init__(self, D: int, name='dimension_reduction_compression', dr_function: incremental_dimension_reduction = None, dr_function_configs: dict = None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.D = D

        if dr_function is not None:
            self.dr_function = dr_function
        elif dr_function_configs is not None:
            function_class = dr_function_configs['function_class']
            function_parameters = dr_function_configs['function_parameters'] if 'function_parameters' in dr_function_configs else {}
            if 'n_feature' in function_parameters:
                assert function_parameters['n_feature'] == D
            else:
                function_parameters['n_feature'] = D
            self.dr_function = config.get_obj_from_str(function_class)(**function_parameters)
        else:
            raise ValueError('You must specify either dr_function or dr_function_configs...')

    def calculate_D(self, m: int):
        assert self.D is not None and self.D <= m, 'You must specify a D that is smaller than m!'
        return self.D

    def forward(self, x: torch.Tensor, device: str = 'cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        compression = self.dr_function(torch.from_numpy(x.numpy())).to(device)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class incremental_PCA_based_compression(dimension_reduction_compression):
    def __init__(self, D: int, name='incremental_PCA_based_compression', *args, **kwargs):
        dr_function = incremental_PCA(n_feature=D)
        super().__init__(D=D, name=name, dr_function=dr_function, *args, **kwargs)


class incremental_random_projection_based_compression(dimension_reduction_compression):
    def __init__(self, D: int, name='incremental_random_projection_based_compression', *args, **kwargs):
        dr_function = incremental_random_projection(n_feature=D)
        super().__init__(D=D, name=name, dr_function=dr_function, *args, **kwargs)




