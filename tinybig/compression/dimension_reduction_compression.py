# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################################
# Dimension Reduction based Compression Function #
##################################################

"""
Dimension reduction based data compression functions.

This module contains the dimension reduction based data compression functions,
including dimension_reduction_compression, incremental_PCA_based_compression, and incremental_random_projection_based_compression.
"""

import torch

from tinybig.compression import transformation
from tinybig.koala.machine_learning.dimension_reduction import incremental_dimension_reduction, incremental_PCA, incremental_random_projection
from tinybig.config.base_config import config


class dimension_reduction_compression(transformation):
    r"""
        The dimension reduction based data compression function.

        This class reduces the dimensionality of input features by applying a specified dimension reduction
        function or initializing it based on provided configurations.

        Notes
        ----------
        Formally, given an input data instance $\mathbf{x} \in {R}^m$, we can represent the feature selection-based data compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \text{dimension-reduction}(\mathbf{x}) \in {R}^d.
            \end{equation}
        $$

        The output dimension $d$ may require manual setup, e.g., as a hyper-parameter $D$.

        Attributes
        ----------
        D : int
            Number of dimensions to retain after compression.
        name : str
            Name of the transformation.
        dr_function : incremental_dimension_reduction
            The dimension reduction function used for compression.

        Parameters
        ----------
        D : int
            Number of dimensions to retain after compression.
        name : str, optional
            Name of the transformation. Defaults to 'dimension_reduction_compression'.
        dr_function : incremental_dimension_reduction, optional
            A pre-configured dimension reduction function. Defaults to None.
        dr_function_configs : dict, optional
            Configuration for initializing the dimension reduction function. Should include the class name
            and optional parameters. Defaults to None.
        *args : tuple
            Additional positional arguments for the parent `transformation` class.
        **kwargs : dict
            Additional keyword arguments for the parent `transformation` class.

        Raises
        ------
        ValueError
            If neither `dr_function` nor `dr_function_configs` are specified.

        Methods
        -------
        __init__(D, name='dimension_reduction_compression', dr_function=None, dr_function_configs=None, *args, **kwargs)
            Initializes the dimension reduction and compression instance.
        calculate_D(m: int)
            Validates and returns the specified number of dimensions (`D`).
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Applies the dimension reduction and compression function to the input tensor.
    """
    def __init__(self, D: int, name='dimension_reduction_compression', dr_function: incremental_dimension_reduction = None, dr_function_configs: dict = None, *args, **kwargs):
        """
            Initializes the dimension reduction and compression instance.

            This method sets the number of dimensions (`D`) to retain and initializes the dimension reduction
            function using either a direct function or a configuration.

            Parameters
            ----------
            D : int
                Number of dimensions to retain after compression.
            name : str, optional
                Name of the transformation. Defaults to 'dimension_reduction_compression'.
            dr_function : incremental_dimension_reduction, optional
                A pre-configured dimension reduction function. Defaults to None.
            dr_function_configs : dict, optional
                Configuration for initializing the dimension reduction function. Should include the class name
                and optional parameters. Defaults to None.
            *args : tuple
                Additional positional arguments for the parent `transformation` class.
            **kwargs : dict
                Additional keyword arguments for the parent `transformation` class.

            Raises
            ------
            ValueError
                If neither `dr_function` nor `dr_function_configs` are specified.

            Returns
            ----------
            transformation
                The feature selection based compression function.
        """
        super().__init__(name=name, *args, **kwargs)
        self.D = D

        if dr_function is not None:
            self.dr_function = dr_function
        elif dr_function_configs is not None:
            function_class = dr_function_configs['function_class']
            function_parameters = dr_function_configs['function_parameters'] if 'function_parameters' in dr_function_configs else {}
            if 'n_feature' in function_parameters:
                assert function_parameters['n_feature'] == D
            else:
                function_parameters['n_feature'] = D
            self.dr_function = config.get_obj_from_str(function_class)(**function_parameters)
        else:
            raise ValueError('You must specify either dr_function or dr_function_configs...')

    def calculate_D(self, m: int):
        """
            The compression dimension calculation method.

            It calculates the intermediate compression space dimension based on the input dimension parameter m.
            This method also validates the specified number of dimensions (`D`) and ensures it is less than or equal to `m`.

            Parameters
            ----------
            m : int
                Total number of features in the input.

            Returns
            -------
            int
                The number of dimensions to retain (`D`).

            Raises
            ------
            AssertionError
                If `D` is not set or is greater than `m`.
        """
        assert self.D is not None and self.D <= m, 'You must specify a D that is smaller than m!'
        return self.D

    def forward(self, x: torch.Tensor, device: str = 'cpu', *args, **kwargs):
        r"""
            The forward method of the dimension reduction based compression function.

            It applies the dimension reduction and compression function to the input tensor.

            Formally, given an input data instance $\mathbf{x} \in {R}^m$, we can represent the feature selection-based data compression function as follows:

            $$
                \begin{equation}
                \kappa(\mathbf{x}) = \text{dimension-reduction}(\mathbf{x}) \in {R}^d.
                \end{equation}
            $$

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, num_features)`.
            device : str, optional
                Device for computation (e.g., 'cpu', 'cuda' or 'mps'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for pre- and post-processing.
            **kwargs : dict
                Additional keyword arguments for pre- and post-processing.

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

        compression = self.dr_function(torch.from_numpy(x.numpy())).to(device)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class incremental_PCA_based_compression(dimension_reduction_compression):
    """
        The incremental PCA dimension reduction based data compression function.

        A dimension reduction and compression class based on incremental PCA.
        This class uses incremental Principal Component Analysis (PCA) to reduce the dimensionality of input features.
        It calls the incremental_PCA method from the tinybig.koala.machine_learning.dimension_reduction.incremental_PCA module.

        Parameters
        ----------
        D : int
            Number of dimensions to retain after compression.
        name : str, optional
            Name of the transformation. Defaults to 'incremental_PCA_based_compression'.
        *args : tuple
            Additional positional arguments for the parent `dimension_reduction_compression` class.
        **kwargs : dict
            Additional keyword arguments for the parent `dimension_reduction_compression` class.


        Methods
        -------
        __init__(D, name='incremental_PCA_based_compression', *args, **kwargs)
            Initializes the class with the incremental PCA method.
    """
    def __init__(self, D: int, name='incremental_PCA_based_compression', *args, **kwargs):
        """
            The incremental PCA dimension reduction based data compression function.

            It initializes the class with the incremental PCA method.

            Parameters
            ----------
            D : int
                Number of dimensions to retain after compression.
            name : str, optional
                Name of the transformation. Defaults to 'incremental_PCA_based_compression'.
            *args : tuple
                Additional positional arguments for the parent `dimension_reduction_compression` class.
            **kwargs : dict
                Additional keyword arguments for the parent `dimension_reduction_compression` class.
        """
        dr_function = incremental_PCA(n_feature=D)
        super().__init__(D=D, name=name, dr_function=dr_function, *args, **kwargs)


class incremental_random_projection_based_compression(dimension_reduction_compression):
    """
        The incremental random projection dimension reduction based data compression function.

        A dimension reduction and compression class based on incremental random projections.
        This class uses incremental random projection methods to reduce the dimensionality of input features.
        It calls the incremental_random_projection from the tinybig.koala.machine_learning.dimension_reduction.incremental_random_projection module.

        Methods
        -------
        __init__(D, name='incremental_random_projection_based_compression', *args, **kwargs)
            Initializes the class with the incremental random projection method.

        Parameters
        ----------
        D : int
            Number of dimensions to retain after compression.
        name : str, optional
            Name of the transformation. Defaults to 'incremental_random_projection_based_compression'.
        *args : tuple
            Additional positional arguments for the parent `dimension_reduction_compression` class.
        **kwargs : dict
            Additional keyword arguments for the parent `dimension_reduction_compression` class.
    """
    def __init__(self, D: int, name='incremental_random_projection_based_compression', *args, **kwargs):
        """
            The incremental random projection dimension reduction based data compression function.

            Initializes the class with the incremental random projection method.

            Parameters
            ----------
            D : int
                Number of dimensions to retain after compression.
            name : str, optional
                Name of the transformation. Defaults to 'incremental_random_projection_based_compression'.
            *args : tuple
                Additional positional arguments for the parent `dimension_reduction_compression` class.
            **kwargs : dict
                Additional keyword arguments for the parent `dimension_reduction_compression` class.
        """
        dr_function = incremental_random_projection(n_feature=D)
        super().__init__(D=D, name=name, dr_function=dr_function, *args, **kwargs)




