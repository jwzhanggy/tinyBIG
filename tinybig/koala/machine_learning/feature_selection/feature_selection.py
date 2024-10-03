
# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################
# Base Feature Selection Method #
#################################

from typing import Union

import numpy as np
import torch

from abc import abstractmethod


class feature_selection(object):

    def __init__(self, name: str = 'feature_selection', n_feature: int = None,  incremental: bool = True, incremental_stop_threshold: float = 0.01, t_threshold: int = 100, *args, **kwargs):
        self.name = name
        self.n_feature = n_feature
        self.incremental = incremental
        self.incremental_stop_threshold = incremental_stop_threshold
        self.t_threshold = t_threshold

    def get_n_feature(self):
        return self.n_feature

    def set_n_feature(self, n_feature):
        self.n_feature = n_feature

    def __call__(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        return self.forward(X=X, device=device, *args, **kwargs)

    def forward(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()
        else:
            input_X = X

        X_reduced = self.fit_transform(X=input_X, device=device, *args, **kwargs)

        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()
        else:
            input_X = X

        self.fit(X=input_X, device=device, *args, **kwargs)
        X_reduced = self.transform(X=input_X, device=device, *args, **kwargs)

        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced

    @abstractmethod
    def fit(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        pass




