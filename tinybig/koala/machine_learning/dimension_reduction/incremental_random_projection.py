
# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################################################
# Incremental Random Projection based Dimension Reduction #
###########################################################

from typing import Union

import numpy as np
import torch

from sklearn.random_projection import SparseRandomProjection

from tinybig.koala.machine_learning.dimension_reduction import incremental_dimension_reduction


class incremental_random_projection(incremental_dimension_reduction):
    """
        Incremental Random Projection for dimensionality reduction.

        This class utilizes `SparseRandomProjection` from `sklearn.random_projection` to perform random projection
        for dimensionality reduction in an incremental manner.

        Attributes
        ----------
        name : str
            The name of the dimension reduction method.
        irp : sklearn.random_projection.SparseRandomProjection
            The underlying SparseRandomProjection model.
        n_feature : int
            The number of components to retain after random projection.

        Methods
        -------
        update_n_feature(new_n_feature: int)
            Update the number of components for the random projection model.
        fit(X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs)
            Fit the random projection model to the input data.
        transform(X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs)
            Transform the input data using the fitted random projection model.
    """
    def __init__(self, name: str = 'incremental_random_projection', *args, **kwargs):
        """
            Initialize the incremental random projection model.

            Parameters
            ----------
            name : str, optional
                The name of the dimension reduction method. Default is 'incremental_random_projection'.
            *args, **kwargs
                Additional arguments passed to the parent class.
        """
        super().__init__(name=name, *args, **kwargs)
        self.irp = SparseRandomProjection(n_components=self.n_feature)

    def update_n_feature(self, new_n_feature: int):
        """
            Update the number of components for the random projection model.

            Parameters
            ----------
            new_n_feature : int
                The new number of components to retain.
        """
        self.set_n_feature(new_n_feature)
        self.irp = SparseRandomProjection(n_components=new_n_feature)

    def fit(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        """
            Fit the random projection model to the input data.

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
        self.irp.fit(input_X)

    def transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        """
            Transform the input data using the fitted random projection model.

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

        X_reduced = self.irp.transform(input_X)

        assert X_reduced.shape[1] == self.n_feature
        return torch.tensor(X_reduced) if isinstance(X, torch.Tensor) and not isinstance(X_reduced, torch.Tensor) else X_reduced




