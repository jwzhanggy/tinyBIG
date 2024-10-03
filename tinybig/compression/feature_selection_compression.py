# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################################
# Feature Selection based Compression Function #
################################################

import torch

from tinybig.compression import transformation
from tinybig.koala.machine_learning.feature_selection import feature_selection, incremental_feature_clustering, incremental_variance_threshold
from tinybig.config.base_config import config


class feature_selection_compression(transformation):
    def __init__(self, D: int, name='feature_selection_compression', fs_function: feature_selection = None, fs_function_configs: dict = None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.D = D

        if fs_function is not None:
            self.fs_function = fs_function
        elif fs_function_configs is not None:
            function_class = fs_function_configs['function_class']
            function_parameters = fs_function_configs['function_parameters'] if 'function_parameters' in fs_function_configs else {}
            if 'n_feature' in function_parameters:
                assert function_parameters['n_feature'] == D
            else:
                function_parameters['n_feature'] = D
            self.fs_function = config.get_obj_from_str(function_class)(**function_parameters)
        else:
            raise ValueError('You must specify either fs_function or fs_function_configs...')

    def calculate_D(self, m: int):
        assert self.D is not None and self.D <= m, 'You must specify a D that is smaller than m!'
        return self.D

    def forward(self, x: torch.Tensor, device: str = 'cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        compression = self.fs_function(torch.from_numpy(x.numpy())).to(device)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class incremental_feature_clustering_based_compression(feature_selection_compression):
    def __init__(self, D: int, name='incremental_feature_clustering_based_compression', *args, **kwargs):
        fs_function = incremental_feature_clustering(n_feature=D)
        super().__init__(D=D, name=name, fs_function=fs_function, *args, **kwargs)


class incremental_variance_threshold_based_compression(feature_selection_compression):
    def __init__(self, D: int, name='incremental_variance_threshold_based_compression', *args, **kwargs):
        fs_function = incremental_variance_threshold(n_feature=D)
        super().__init__(D=D, name=name, fs_function=fs_function, *args, **kwargs)




