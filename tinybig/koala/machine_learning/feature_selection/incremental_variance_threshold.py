
# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################################################
# Incremental Variance Threshold based Feature Selection #
##########################################################

from typing import Union

import numpy as np
import torch

from tinybig.koala.machine_learning.feature_selection import feature_selection
from tinybig.koala.statistics import batch_variance
from tinybig.koala.linear_algebra import euclidean_distance


class incremental_variance_threshold(feature_selection):
    """
        Incremental variance-based feature selection.

        This class selects features based on their variance, either by applying a threshold or selecting a fixed number
        of features with the highest variance. It supports incremental updates to the variance estimates.

        Attributes
        ----------
        threshold : float
            The minimum variance threshold for feature selection.
        v : torch.Tensor or None
            The current variance estimates for each feature.
        t : int or None
            The iteration count for incremental updates.

        Methods
        -------
        update_n_feature(new_n_feature)
            Update the number of features to select.
        update_threshold(new_threshold)
            Update the variance threshold for feature selection.
        update_v(new_v)
            Incrementally update the variance estimates.
        fit(X, device='cpu', *args, **kwargs)
            Compute variance estimates for the input data.
        transform(X, device='cpu', *args, **kwargs)
            Select features based on the variance estimates.
    """
    def __init__(self, threshold: float = 0.0, name: str = 'incremental_variance_threshold', *args, **kwargs):
        """
            Initialize the incremental variance threshold feature selection class.

            Parameters
            ----------
            threshold : float, optional
                The minimum variance threshold for feature selection. Default is 0.0.
            name : str, optional
                The name of the feature selection method. Default is 'incremental_variance_threshold'.
            *args, **kwargs
                Additional arguments for the base class.
        """
        super().__init__(name=name, *args, **kwargs)

        self.threshold = threshold
        self.v = None
        self.t = None

    def update_n_feature(self, new_n_feature: int):
        """
            Update the number of features to select.

            Parameters
            ----------
            new_n_feature : int
                The new number of features to select.
        """
        assert new_n_feature > 0
        self.set_n_feature(new_n_feature)
        self.v = None
        self.t = None

    def update_threshold(self, new_threshold: float):
        """
            Update the variance threshold for feature selection.

            Parameters
            ----------
            new_threshold : float
                The new variance threshold.
        """
        self.threshold = new_threshold
        self.v = None
        self.t = None

    def update_v(self, new_v: torch.Tensor):
        """
            Incrementally update the variance estimates.

            Parameters
            ----------
            new_v : torch.Tensor
                The new variance estimates to update or replace the current estimates.
        """
        if self.incremental:
            if self.v is None:
                self.v = torch.zeros_like(new_v)
                self.t = 0

            assert new_v.shape == self.v.shape and self.t >= 0
            self.t += 1
            old_v = self.v
            self.v = ((self.t - 1) * self.v + new_v)/self.t

            if self.t >= self.t_threshold or euclidean_distance(x=old_v, x2=self.v) < self.incremental_stop_threshold:
                self.incremental = False
        else:
            self.v = new_v

    def fit(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        """
            Compute variance estimates for the input data.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data for feature selection.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            *args, **kwargs
                Additional arguments for the fitting process.
        """
        X = torch.tensor(X)
        new_v = batch_variance(X, dim=0)
        self.update_v(new_v)

    def transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        """
            Select features based on the variance estimates.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data to transform.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            *args, **kwargs
                Additional arguments for the transformation process.

            Returns
            -------
            Union[np.ndarray, torch.Tensor]
                The input data with selected features.
        """
        input_X = torch.tensor(X)

        assert self.v is not None and self.v.shape[0] == input_X.shape[1]

        if self.n_feature is not None:
            n = min(self.n_feature, input_X.shape[1])
            indices = np.argsort(self.v)[-n:]
        else:
            indices = np.where(self.v >= self.threshold)[0]

        if len(indices) == 0:
            indices = np.arange(self.v.size)

        X_selected = input_X[:, indices]

        assert X_selected.shape[1] == self.n_feature
        return X_selected.detach().cpu().numpy() if isinstance(X, np.ndarray) and not isinstance(X_selected, np.ndarray) else X_selected




