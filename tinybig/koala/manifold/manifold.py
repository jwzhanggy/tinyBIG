# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############
# Manifolds #
#############

from typing import Union

import torch
import numpy as np

from abc import abstractmethod

from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS, SpectralEmbedding, smacof


class manifold(object):

    def __init__(self, name: str = 'base_manifold', n_neighbors: int = 5, n_components: int = 2, *args, **kwargs):
        self.name = name
        self.n_neighbors = n_neighbors
        self.n_components = n_components

        self.model = None
        self.init_model()

    def get_n_neighbors(self):
        return self.n_neighbors

    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.init_model()

    def get_n_components(self):
        return self.n_components

    def set_n_components(self, n_components):
        self.n_components = n_components
        self.init_model()

    def __call__(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        return self.forward(X=X, device=device, *args, **kwargs)

    def forward(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        return self.fit_transform(X=X, device=device, *args, **kwargs)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
        else:
            input_X = X

        X_manifold = self.model.fit_transform(X=input_X)

        return torch.tensor(X_manifold) if isinstance(X, torch.Tensor) and not isinstance(X_manifold, torch.Tensor) else X_manifold

    @abstractmethod
    def init_model(self):
        pass


class isomap_manifold(manifold):
    def __init__(self, name: str = 'isomap_manifold', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        self.model = Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components)


class lle_manifold(manifold):
    def __init__(self, name: str = 'lle_manifold', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        self.model = LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components)


class mds_manifold(manifold):
    def __init__(self, name: str = 'mds_manifold', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        self.model = MDS(n_components=self.n_components)


class spectral_embedding_manifold(manifold):
    def __init__(self, name: str = 'spectral_embedding_manifold', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        self.model = SpectralEmbedding(n_components=self.n_components)


class tsne_manifold(manifold):
    def __init__(self, perplexity: float, name: str = 'tsne_manifold', *args, **kwargs):
        self.perplexity = perplexity
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        self.model = TSNE(n_components=self.n_components, perplexity=self.perplexity)


