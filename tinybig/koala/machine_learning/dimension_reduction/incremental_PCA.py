
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
    """
        Incremental Principal Component Analysis (PCA) for dimensionality reduction.

        This class leverages `IncrementalPCA` from `sklearn.decomposition` to perform PCA in an incremental manner,
        enabling efficient processing of large datasets that cannot fit into memory.

        Attributes
        ----------
        name : str
            The name of the dimension reduction method.
        ipca : sklearn.decomposition.IncrementalPCA
            The underlying incremental PCA model.
        n_feature : int
            The number of components to retain after PCA.

        Methods
        -------
        update_n_feature(new_n_feature: int)
            Update the number of components for the PCA model.
        fit(X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs)
            Fit the incremental PCA model to the input data.
        transform(X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs)
            Transform the input data using the fitted PCA model.
    """
    def __init__(self, name: str = 'incremental_PCA', *args, **kwargs):
        """
            Initialize the incremental PCA model.

            Parameters
            ----------
            name : str, optional
                The name of the dimension reduction method. Default is 'incremental_PCA'.
            *args, **kwargs
                Additional arguments passed to the parent class.
        """
        super().__init__(name=name, *args, **kwargs)
        self.ipca = IncrementalPCA(n_components=self.n_feature)

    def update_n_feature(self, new_n_feature: int):
        """
            Update the number of components for the PCA model.

            Parameters
            ----------
            new_n_feature : int
                The new number of components to retain.
        """
        self.set_n_feature(new_n_feature)
        self.ipca = IncrementalPCA(n_components=new_n_feature)

    def fit(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        """
            Fit the incremental PCA model to the input data.

            Parameters
            ----------
            X : Union[np.ndarray, torch.Tensor]
                The input data for fitting.
            device : str, optional
                The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            *args, **kwargs
                Additional arguments for the fit process.
        """
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
        else:
            input_X = X
        self.ipca.partial_fit(input_X)

    def transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        """
            Transform the input data using the fitted PCA model.

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
                The transformed data after dimensionality reduction.
        """
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
        else:
            input_X = X
        assert self.n_feature is not None and 0 < self.n_feature <= X.shape[1]

        X_reduced = self.ipca.transform(input_X)

        assert X_reduced.shape[1] == self.n_feature
        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced






