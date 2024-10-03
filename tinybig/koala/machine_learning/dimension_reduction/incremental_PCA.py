
# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################################
# Incremental PCA Dimension Reduction #
#######################################

from typing import Union

import numpy as np
import torch

from sklearn.decomposition import IncrementalPCA

from tinybig.koala.machine_learning.dimension_reduction import incremental_dimension_reduction


class incremental_PCA(incremental_dimension_reduction):
    def __init__(self, name: str = 'incremental_PCA', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.ipca = IncrementalPCA(n_components=self.n_feature)

    def update_n_feature(self, new_n_feature: int):
        self.set_n_feature(new_n_feature)
        self.ipca = IncrementalPCA(n_components=new_n_feature)

    def fit(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
        else:
            input_X = X
        self.ipca.partial_fit(input_X)

    def transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
        else:
            input_X = X
        assert self.n_feature is not None and 0 < self.n_feature <= X.shape[1]

        X_reduced = self.ipca.transform(input_X)

        assert X_reduced.shape[1] == self.n_feature
        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced






