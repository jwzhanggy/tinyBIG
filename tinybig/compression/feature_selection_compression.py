# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################################
# Feature Selection based Compression Function #
################################################

"""
Feature selection based data compression functions.

This module contains the feature selection based data compression functions, including
    feature_selection_compression,
    incremental_feature_clustering_based_compression,
    incremental_variance_threshold_based_compression.
"""

import torch

from tinybig.compression import transformation
from tinybig.koala.machine_learning.feature_selection import (
    feature_selection,
    incremental_feature_clustering,
    incremental_variance_threshold
)
from tinybig.config.base_config import config


class feature_selection_compression(transformation):
    r"""
        The feature selection based data compression function.

        It performs the data compression with the provided feature selection method,
        which incrementally select useful features from the provided data batch.

        Notes
        ----------
        Formally, given an input data instance $\mathbf{x} \in {R}^m$, we can represent the feature selection-based data compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \text{feature-selection}(\mathbf{x}) \in {R}^d.
            \end{equation}
        $$

        The output dimension $d$ may require manual setup, e.g., as a hyper-parameter $D$.
        
        Parameters
        ----------
        D : int
            Number of features to retain after compression.
        name : str, default = 'feature_selection_compression'
            Name of the transformation.
        fs_function : feature_selection, default = None
            A pre-configured feature selection function.
        fs_function_configs : dict, default = None
            Configuration for initializing the feature selection function. Should include the class name
            and optional parameters.

        Raises
        ------
        ValueError
            If neither `fs_function` nor `fs_function_configs` are specified.

        Methods
        -------
        __init__(D, name='feature_selection_compression', fs_function=None, fs_function_configs=None, *args, **kwargs)
            Initializes the feature selection based compression function.
        calculate_D(m: int)
            It validates and returns the specified number of features (`D`).
        forward(x: torch.Tensor, device: str = 'cpu', *args, **kwargs)
            It applies the feature selection and compression function to the input tensor.
    """
    def __init__(self, D: int, name='feature_selection_compression', fs_function: feature_selection = None, fs_function_configs: dict = None, *args, **kwargs):
        """
            The initialization method of the feature selection based compression function.

            It initializes the compression function based on
            the provided feature selection method (or its configs).

            Parameters
            ----------
            D : int
                Number of features to retain after compression.
            name : str, default = 'feature_selection_compression'
                Name of the transformation.
            fs_function : feature_selection, default = None
                A pre-configured feature selection function.
            fs_function_configs : dict, default = None
                Configuration for initializing the feature selection function. Should include the class name
                and optional parameters.

            Returns
            ----------
            transformation
                The feature selection based compression function.
        """

        super().__init__(name=name, *args, **kwargs)
        self.D = D

        if fs_function is not None:
            self.fs_function = fs_function
        elif fs_function_configs is not None:
            function_class = fs_function_configs['function_class']
            function_parameters = fs_function_configs['function_parameters'] if 'function_parameters' in fs_function_configs else {}
            if 'n_feature' in function_parameters:
                assert function_parameters['n_feature'] == D
            else:
                function_parameters['n_feature'] = D
            self.fs_function = config.get_obj_from_str(function_class)(**function_parameters)
        else:
            raise ValueError('You must specify either fs_function or fs_function_configs...')

    def calculate_D(self, m: int):
        """
            The compression dimension calculation method.

            It calculates the intermediate compression space dimension based on the input dimension parameter m.
            This method also validates the specified number of features (`D`) and ensures it is less than or equal to `m`.

            Parameters
            ----------
            m : int
                Total number of features in the input.

            Returns
            -------
            int
                The number of features to retain (`D`).

            Raises
            ------
            AssertionError
                If `D` is not set or is greater than `m`.
        """
        assert self.D is not None and self.D <= m, 'You must specify a D that is smaller than m!'
        return self.D

    def forward(self, x: torch.Tensor, device: str = 'cpu', *args, **kwargs):
        r"""
            The forward method of the feature selection based compression function.

            It applies the feature selection based compression function to the input tensor.

            Formally, given an input data instance $\mathbf{x} \in {R}^m$, we can represent the feature selection-based data compression function as follows:

            $$
                \begin{equation}
                \kappa(\mathbf{x}) = \text{feature-selection}(\mathbf{x}) \in {R}^d.
                \end{equation}
            $$

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, num_features)`.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

            Returns
            -------
            torch.Tensor
                Compressed tensor of shape `(batch_size, D)`.

            Raises
            ------
            AssertionError
                If the output tensor shape does not match the expected `(batch_size, D)`.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        compression = self.fs_function(torch.from_numpy(x.numpy())).to(device)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class incremental_feature_clustering_based_compression(feature_selection_compression):
    """
        The incremental feature clustering based compression function.

        This class uses an incremental feature clustering method to reduce the dimensionality of input features for data compression.
        It calls the incremental_feature_clustering method from tinybig.compression.incremental_feature_clustering module.

        Parameters
        ----------
        D : int
            Number of features to retain after compression.
        name : str, optional
            Name of the transformation. Defaults to 'incremental_feature_clustering_based_compression'.

        Methods
        -------
        __init__(D, name='incremental_feature_clustering_based_compression', *args, **kwargs)
            Initializes the class with the incremental feature clustering method.
    """

    def __init__(self, D: int, name='incremental_feature_clustering_based_compression', *args, **kwargs):
        """
            The initialization method incremental feature clustering based compression function.

            It initializes the class with the incremental feature clustering method.

            Parameters
            ----------
            D : int
                Number of features to retain after compression.
            name : str, optional
                Name of the transformation. Defaults to 'incremental_feature_clustering_based_compression'.
            """
        fs_function = incremental_feature_clustering(n_feature=D)
        super().__init__(D=D, name=name, fs_function=fs_function, *args, **kwargs)


class incremental_variance_threshold_based_compression(feature_selection_compression):
    """
        The incremental feature clustering based compression function.

        This class uses an incremental variance thresholding method to reduce the dimensionality of input features.
        It calls the incremental_variance_threshold from the tinybig.compression.incremental_variance_threshold module.

        Parameters
        ----------
        D : int
            Number of features to retain after compression.
        name : str, optional
            Name of the transformation. Defaults to 'incremental_variance_threshold_based_compression'.

        Methods
        -------
        __init__(D, name='incremental_variance_threshold_based_compression', *args, **kwargs)
            Initializes the class with the incremental variance thresholding method.
    """

    def __init__(self, D: int, name='incremental_variance_threshold_based_compression', *args, **kwargs):
        """
            The initialization method incremental variance thresholding based compression function.

            It initializes the class with the incremental variance thresholding method.

            Parameters
            ----------
            D : int
                Number of features to retain after compression.
            name : str, optional
                Name of the transformation. Defaults to 'incremental_variance_threshold_based_compression'.
        """
        fs_function = incremental_variance_threshold(n_feature=D)
        super().__init__(D=D, name=name, fs_function=fs_function, *args, **kwargs)




