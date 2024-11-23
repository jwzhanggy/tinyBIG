
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
    """
        A base class for incremental dimension reduction methods.

        This class provides a framework for implementing incremental dimension reduction techniques,
        with methods for setting the number of features, fitting, transforming, and reducing data.

        Attributes
        ----------
        name : str
            The name of the dimension reduction method.
        n_feature : int, optional
            The number of features to retain after reduction.
        incremental : bool, optional
            Whether the dimension reduction is performed incrementally.

        Methods
        -------
        get_n_feature()
            Retrieve the number of features to retain.
        set_n_feature(n_feature)
            Set the number of features to retain.
        __call__(X, device='cup', *args, **kwargs)
            Forward the input data through the dimension reduction process.
        forward(X, device='cup', *args, **kwargs)
            Perform dimension reduction on the input data.
        fit_transform(X, device='cup', *args, **kwargs)
            Fit the model to the input data and reduce its dimensionality.
        fit(X, device='cup', *args, **kwargs)
            Abstract method for fitting the model to the input data.
        transform(X, device='cup', *args, **kwargs)
            Abstract method for transforming input data based on the fitted model.
    """
    def __init__(self, name: str = 'incremental_dimension_reduction', n_feature: int = None, incremental: bool = True, *args, **kwargs):
        """
            Initialize the incremental dimension reduction class.

            Parameters
            ----------
            name : str, optional
                The name of the dimension reduction method. Default is 'incremental_dimension_reduction'.
            n_feature : int, optional
                The number of features to retain after reduction. Default is None.
            incremental : bool, optional
                Whether the dimension reduction is performed incrementally. Default is True.
            *args, **kwargs
                Additional arguments for subclass initialization.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.incremental = incremental
        self.n_feature = n_feature

    def get_n_feature(self):
        """
            Retrieve the number of features to retain.

            Returns
            -------
            int
                The number of features to retain.
        """
        return self.n_feature

    def set_n_feature(self, n_feature):
        """
            Set the number of features to retain.

            Parameters
            ----------
            n_feature : int
                The number of features to retain.
        """
        self.n_feature = n_feature

    def __call__(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Forward the input data through the dimension reduction process.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data to reduce.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cup'.
            *args, **kwargs
                Additional arguments for the forward process.

            Returns
            -------
            Union[np.ndarray, torch.Tensor]
                The reduced input data.
        """
        return self.forward(X=X, device=device, *args, **kwargs)

    def forward(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Perform dimension reduction on the input data.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data to reduce.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cup'.
            *args, **kwargs
                Additional arguments for the forward process.

            Returns
            -------
            Union[np.ndarray, torch.Tensor]
                The reduced input data.
        """
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
        else:
            input_X = X
        X_reduced = self.fit_transform(X=input_X, device=device, *args, **kwargs)
        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Fit the model to the input data and reduce its dimensionality.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data to reduce.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cup'.
            *args, **kwargs
                Additional arguments for the fit and transform processes.

            Returns
            -------
            Union[np.ndarray, torch.Tensor]
                The reduced input data.
        """
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
        else:
            input_X = X

        self.fit(X=input_X, device=device, *args, **kwargs)
        X_reduced = self.transform(X=input_X, device=device, *args, **kwargs)

        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced


    @abstractmethod
    def fit(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Abstract method for fitting the model to the input data.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data to fit.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cup'.
            *args, **kwargs
                Additional arguments for the fitting process.

            Raises
            ------
            NotImplementedError
                This method must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
            Abstract method for transforming input data based on the fitted model.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data to transform.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cup'.
            *args, **kwargs
                Additional arguments for the transformation process.

            Raises
            ------
            NotImplementedError
                This method must be implemented in subclasses.
        """
        pass




