
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


class incremental_dimension_reduction(object):

    def __init__(self, name: str = 'incremental_dimension_reduction', n_feature: int = None, incremental: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.incremental = incremental
        self.n_feature = n_feature

    def get_n_feature(self):
        return self.n_feature

    def set_n_feature(self, n_feature):
        self.n_feature = n_feature

    def __call__(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        return self.forward(X=X, device=device, *args, **kwargs)

    def forward(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
        else:
            input_X = X
        X_reduced = self.fit_transform(X=input_X, device=device, *args, **kwargs)
        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
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




