
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
    """
        Base class for feature selection.

        This class provides an abstract base for implementing feature selection algorithms. It supports
        incremental feature selection with stopping criteria based on thresholds.

        Attributes
        ----------
        name : str
            The name of the feature selection method.
        n_feature : int
            The number of features to select.
        incremental : bool
            Whether to perform incremental feature selection.
        incremental_stop_threshold : float
            The threshold for stopping incremental feature selection.
        t_threshold : int
            A time or iteration threshold for incremental selection.

        Methods
        -------
        get_n_feature()
            Get the number of features to select.
        set_n_feature(n_feature)
            Set the number of features to select.
        forward(X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs)
            Apply feature selection and return the reduced features.
        fit_transform(X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs)
            Fit the feature selection model and return the reduced features.
        fit(X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs)
            Abstract method to fit the feature selection model to the input data.
        transform(X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs)
            Abstract method to transform the input data using the fitted model.
    """

    def __init__(self, name: str = 'feature_selection', n_feature: int = None,  incremental: bool = True, incremental_stop_threshold: float = 0.01, t_threshold: int = 100, *args, **kwargs):
        """
            Initialize the feature selection class.

            Parameters
            ----------
            name : str, optional
                The name of the feature selection method. Default is 'feature_selection'.
            n_feature : int, optional
                The number of features to select. Default is None.
            incremental : bool, optional
                Whether to perform incremental feature selection. Default is True.
            incremental_stop_threshold : float, optional
                The threshold for stopping incremental feature selection. Default is 0.01.
            t_threshold : int, optional
                A time or iteration threshold for incremental selection. Default is 100.
            *args, **kwargs
                Additional arguments for customization.
        """

        self.name = name
        self.n_feature = n_feature
        self.incremental = incremental
        self.incremental_stop_threshold = incremental_stop_threshold
        self.t_threshold = t_threshold

    def get_n_feature(self):
        """
            Get the number of features to select.

            Returns
            -------
            int
                The number of features to select.
        """
        return self.n_feature

    def set_n_feature(self, n_feature):
        """
            Get the number of features to select.

            Returns
            -------
            int
                The number of features to select.
        """
        self.n_feature = n_feature

    def __call__(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Apply the feature selection model.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data for feature selection.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            *args, **kwargs
                Additional arguments for the feature selection process.

            Returns
            -------
            Union[np.ndarray, torch.Tensor]
                The reduced features after selection.
        """
        return self.forward(X=X, device=device, *args, **kwargs)

    def forward(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Apply feature selection and return the reduced features.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data for feature selection.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            *args, **kwargs
                Additional arguments for the feature selection process.

            Returns
            -------
            Union[np.ndarray, torch.Tensor]
                The reduced features after selection.
        """
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()
        else:
            input_X = X

        X_reduced = self.fit_transform(X=input_X, device=device, *args, **kwargs)

        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Fit the feature selection model and return the reduced features.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data for fitting and transformation.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            *args, **kwargs
                Additional arguments for the feature selection process.

            Returns
            -------
            Union[np.ndarray, torch.Tensor]
                The reduced features after fitting and transformation.
        """
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()
        else:
            input_X = X

        self.fit(X=input_X, device=device, *args, **kwargs)
        X_reduced = self.transform(X=input_X, device=device, *args, **kwargs)

        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced

    @abstractmethod
    def fit(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Fit the feature selection model to the input data.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data for fitting.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            *args, **kwargs
                Additional arguments for fitting the feature selection model.
        """
        pass

    @abstractmethod
    def transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Transform the input data using the fitted feature selection model.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data to transform.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            *args, **kwargs
                Additional arguments for the transformation process.
        """
        pass




