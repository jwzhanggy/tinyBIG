# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################################
# Manifold based Compression Function #
#######################################

import torch

from tinybig.compression import transformation
from tinybig.koala.manifold import manifold, isomap_manifold, lle_manifold, mds_manifold, spectral_embedding_manifold, tsne_manifold
from tinybig.config.base_config import config


class manifold_compression(transformation):
    def __init__(self, D: int, n_neighbors: int = 1, name='dimension_reduction_compression', manifold_function: manifold = None, manifold_function_configs: dict = None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.D = D
        self.n_neighbors = n_neighbors

        if manifold_function is not None:
            self.manifold_function = manifold_function
        elif manifold_function_configs is not None:
            function_class = manifold_function_configs['function_class']
            function_parameters = manifold_function_configs['function_parameters'] if 'function_parameters' in manifold_function_configs else {}
            if 'n_components' in function_parameters:
                assert function_parameters['n_components'] == D
            else:
                function_parameters['n_components'] = D
            self.manifold_function = config.get_obj_from_str(function_class)(**function_parameters)
        else:
            raise ValueError('You must specify either manifold_function or manifold_function_configs...')

    def calculate_D(self, m: int):
        return self.D

    def forward(self, x: torch.Tensor, device: str = 'cpu', *args, **kwargs):
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        compression = self.manifold_function(torch.from_numpy(x.numpy())).to(device)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class isomap_manifold_compression(manifold_compression):
    def __init__(self, D: int, n_neighbors: int = 1, name='isomap_manifold_compression', *args, **kwargs):
        manifold_function = isomap_manifold(n_components=D, n_neighbors=n_neighbors)
        super().__init__(D=D, name=name, manifold_function=manifold_function, *args, **kwargs)


class lle_manifold_compression(manifold_compression):
    def __init__(self, D: int, n_neighbors: int = 1, name='lle_manifold_compression', *args, **kwargs):
        manifold_function = lle_manifold(n_components=D, n_neighbors=n_neighbors)
        super().__init__(D=D, n_neighbors=n_neighbors, name=name, manifold_function=manifold_function, *args, **kwargs)


class mds_manifold_compression(manifold_compression):
    def __init__(self, D: int, name='mds_manifold_compression', *args, **kwargs):
        manifold_function = mds_manifold(n_components=D)
        super().__init__(D=D, name=name, manifold_function=manifold_function, *args, **kwargs)


class spectral_embedding_manifold_compression(manifold_compression):
    def __init__(self, D: int, name='spectral_embedding_manifold_compression', *args, **kwargs):
        manifold_function = spectral_embedding_manifold(n_components=D)
        super().__init__(D=D, name=name, manifold_function=manifold_function, *args, **kwargs)


class tsne_manifold_compression(manifold_compression):
    def __init__(self, D: int, perplexity: float, name='tsne_manifold_compression', *args, **kwargs):
        manifold_function = tsne_manifold(n_components=D, perplexity=perplexity)
        super().__init__(D=D, name=name, manifold_function=manifold_function, *args, **kwargs)
