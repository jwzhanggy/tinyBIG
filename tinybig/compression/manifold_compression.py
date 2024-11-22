# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################################
# Manifold based Compression Function #
#######################################
"""
Manifold based data compression functions.

This module contains the manifold based data compression functions,
including manifold_compression, isomap_manifold_compression, lle_manifold_compression,
mds_manifold_compression, tsne_manifold_compression, and spectral_embedding_manifold_compression,
"""

import torch

from tinybig.compression import transformation
from tinybig.koala.manifold import manifold, isomap_manifold, lle_manifold, mds_manifold, spectral_embedding_manifold, tsne_manifold
from tinybig.config.base_config import config


class manifold_compression(transformation):
    r"""
        The manifold based data compression function.

        This class reduces the dimensionality of input features by applying a manifold learning technique
        such as Isomap, Locally Linear Embedding (LLE), or other manifold methods.

        Notes
        ----------
        Formally, given an input data instance $\mathbf{x} \in {R}^m$, we can represent the feature selection-based data compression function as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x}) = \text{manifold}(\mathbf{x}) \in {R}^d.
            \end{equation}
        $$

        The output dimension $d$ may require manual setup, e.g., as a hyper-parameter $D$.

        Attributes
        ----------
        D : int
            Number of dimensions to retain after compression.
        n_neighbors : int
            Number of neighbors used in the manifold learning algorithm.
        name : str
            Name of the transformation.
        manifold_function : manifold
            The manifold learning function used for compression.

        Parameters
        ----------
        D : int
            Number of dimensions to retain after compression.
        n_neighbors : int, optional
            Number of neighbors to use for manifold learning. Defaults to 1.
        name : str, optional
            Name of the transformation. Defaults to 'dimension_reduction_compression'.
        manifold_function : manifold, optional
            A pre-configured manifold function. Defaults to None.
        manifold_function_configs : dict, optional
            Configuration for initializing the manifold function. Should include the class name
            and optional parameters. Defaults to None.
        *args : tuple
            Additional positional arguments for the parent `transformation` class.
        **kwargs : dict
            Additional keyword arguments for the parent `transformation` class.

        Raises
        ------
        ValueError
            If neither `manifold_function` nor `manifold_function_configs` are specified.

        Methods
        -------
        __init__(D, n_neighbors=1, name='dimension_reduction_compression', manifold_function=None, manifold_function_configs=None, *args, **kwargs)
            Initializes the manifold-based dimensionality reduction instance.
        calculate_D(m: int)
            Returns the number of dimensions to retain (`D`).
        forward(x: torch.Tensor, device='cpu', *args, **kwargs)
            Applies the manifold function to the input tensor and reduces its dimensionality.
    """
    def __init__(self, D: int, n_neighbors: int = 1, name='dimension_reduction_compression', manifold_function: manifold = None, manifold_function_configs: dict = None, *args, **kwargs):
        """
            The initialization method of the manifold based compression function.

            It initializes the compression function based on the provided manifold function or its configs.

            Parameters
            ----------
            D : int
                Number of dimensions to retain after compression.
            n_neighbors : int, optional
                Number of neighbors to use for manifold learning. Defaults to 1.
            name : str, optional
                Name of the transformation. Defaults to 'dimension_reduction_compression'.
            manifold_function : manifold, optional
                A pre-configured manifold function. Defaults to None.
            manifold_function_configs : dict, optional
                Configuration for initializing the manifold function. Should include the class name
                and optional parameters. Defaults to None.
            *args : tuple
                Additional positional arguments for the parent `transformation` class.
            **kwargs : dict
                Additional keyword arguments for the parent `transformation` class.

            Raises
            ------
            ValueError
                If neither `manifold_function` nor `manifold_function_configs` are specified.
        """
        super().__init__(name=name, *args, **kwargs)
        self.D = D
        self.n_neighbors = n_neighbors

        if manifold_function is not None:
            self.manifold_function = manifold_function
        elif manifold_function_configs is not None:
            function_class = manifold_function_configs['function_class']
            function_parameters = manifold_function_configs['function_parameters'] if 'function_parameters' in manifold_function_configs else {}
            if 'n_components' in function_parameters:
                assert function_parameters['n_components'] == D
            else:
                function_parameters['n_components'] = D
            self.manifold_function = config.get_obj_from_str(function_class)(**function_parameters)
        else:
            raise ValueError('You must specify either manifold_function or manifold_function_configs...')

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
            The forward method of the manifold based compression function.

            It applies the manifold function to the input tensor and reduces its dimensionality.

            Formally, given an input data instance $\mathbf{x} \in {R}^m$, we can represent the feature selection-based data compression function as follows:

            $$
                \begin{equation}
                \kappa(\mathbf{x}) = \text{manifold}(\mathbf{x}) \in {R}^d.
                \end{equation}
            $$

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, num_features)`.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
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

        compression = self.manifold_function(torch.from_numpy(x.numpy())).to(device)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class isomap_manifold_compression(manifold_compression):
    """
        A manifold-based dimensionality reduction class using Isomap.

        This class applies the Isomap algorithm to reduce the dimensionality of input data,
        preserving the global geometric structure of the data.

        Methods
        -------
        __init__(D, n_neighbors=1, name='isomap_manifold_compression', *args, **kwargs)
            Initializes the Isomap-based dimensionality reduction instance.

        Parameters
        ----------
        D : int
            Number of dimensions to retain after compression.
        n_neighbors : int, optional
            Number of neighbors to use in the Isomap algorithm. Defaults to 1.
        name : str, optional
            Name of the transformation. Defaults to 'isomap_manifold_compression'.
        *args : tuple
            Additional positional arguments for the parent `manifold_compression` class.
        **kwargs : dict
            Additional keyword arguments for the parent `manifold_compression` class.
    """
    def __init__(self, D: int, n_neighbors: int = 1, name='isomap_manifold_compression', *args, **kwargs):
        """
            Initializes the Isomap-based dimensionality reduction instance.

            Parameters
            ----------
            D : int
                Number of dimensions to retain after compression.
            n_neighbors : int, optional
                Number of neighbors to use in the Isomap algorithm. Defaults to 1.
            name : str, optional
                Name of the transformation. Defaults to 'isomap_manifold_compression'.
            *args : tuple
                Additional positional arguments for the parent `manifold_compression` class.
            **kwargs : dict
                Additional keyword arguments for the parent `manifold_compression` class.
        """
        manifold_function = isomap_manifold(n_components=D, n_neighbors=n_neighbors)
        super().__init__(D=D, name=name, manifold_function=manifold_function, *args, **kwargs)


class lle_manifold_compression(manifold_compression):
    """
        A manifold-based dimensionality reduction class using Locally Linear Embedding (LLE).

        This class applies the LLE algorithm to reduce the dimensionality of input data,
        preserving local neighborhood relationships.

        Methods
        -------
        __init__(D, n_neighbors=1, name='lle_manifold_compression', *args, **kwargs)
            Initializes the LLE-based dimensionality reduction instance.

        Parameters
        ----------
        D : int
            Number of dimensions to retain after compression.
        n_neighbors : int, optional
            Number of neighbors to use in the LLE algorithm. Defaults to 1.
        name : str, optional
            Name of the transformation. Defaults to 'lle_manifold_compression'.
        *args : tuple
            Additional positional arguments for the parent `manifold_compression` class.
        **kwargs : dict
            Additional keyword arguments for the parent `manifold_compression` class.
    """
    def __init__(self, D: int, n_neighbors: int = 1, name='lle_manifold_compression', *args, **kwargs):
        """
            Initializes the LLE-based dimensionality reduction instance.

            Parameters
            ----------
            D : int
                Number of dimensions to retain after compression.
            n_neighbors : int, optional
                Number of neighbors to use in the LLE algorithm. Defaults to 1.
            name : str, optional
                Name of the transformation. Defaults to 'lle_manifold_compression'.
            *args : tuple
                Additional positional arguments for the parent `manifold_compression` class.
            **kwargs : dict
                Additional keyword arguments for the parent `manifold_compression` class.
        """
        manifold_function = lle_manifold(n_components=D, n_neighbors=n_neighbors)
        super().__init__(D=D, n_neighbors=n_neighbors, name=name, manifold_function=manifold_function, *args, **kwargs)


class mds_manifold_compression(manifold_compression):
    """
        A manifold-based dimensionality reduction class using Multidimensional Scaling (MDS).

        This class applies the MDS algorithm to reduce the dimensionality of input data,
        preserving pairwise distances between points.

        Methods
        -------
        __init__(D, name='mds_manifold_compression', *args, **kwargs)
            Initializes the MDS-based dimensionality reduction instance.

        Parameters
        ----------
        D : int
            Number of dimensions to retain after compression.
        name : str, optional
            Name of the transformation. Defaults to 'mds_manifold_compression'.
        *args : tuple
            Additional positional arguments for the parent `manifold_compression` class.
        **kwargs : dict
            Additional keyword arguments for the parent `manifold_compression` class.
    """
    def __init__(self, D: int, name='mds_manifold_compression', *args, **kwargs):
        """
            Initializes the MDS-based dimensionality reduction instance.

            Parameters
            ----------
            D : int
                Number of dimensions to retain after compression.
            name : str, optional
                Name of the transformation. Defaults to 'mds_manifold_compression'.
            *args : tuple
                Additional positional arguments for the parent `manifold_compression` class.
            **kwargs : dict
                Additional keyword arguments for the parent `manifold_compression` class.
        """
        manifold_function = mds_manifold(n_components=D)
        super().__init__(D=D, name=name, manifold_function=manifold_function, *args, **kwargs)


class spectral_embedding_manifold_compression(manifold_compression):
    """
        A manifold-based dimensionality reduction class using Spectral Embedding.

        This class applies the Spectral Embedding algorithm to reduce the dimensionality of input data,
        preserving the structure of data represented as a graph.

        Methods
        -------
        __init__(D, name='spectral_embedding_manifold_compression', *args, **kwargs)
            Initializes the Spectral Embedding-based dimensionality reduction instance.

        Parameters
        ----------
        D : int
            Number of dimensions to retain after compression.
        name : str, optional
            Name of the transformation. Defaults to 'spectral_embedding_manifold_compression'.
        *args : tuple
            Additional positional arguments for the parent `manifold_compression` class.
        **kwargs : dict
            Additional keyword arguments for the parent `manifold_compression` class.
    """
    def __init__(self, D: int, name='spectral_embedding_manifold_compression', *args, **kwargs):
        """
            Initializes the Spectral Embedding-based dimensionality reduction instance.

            Parameters
            ----------
            D : int
                Number of dimensions to retain after compression.
            name : str, optional
                Name of the transformation. Defaults to 'spectral_embedding_manifold_compression'.
            *args : tuple
                Additional positional arguments for the parent `manifold_compression` class.
            **kwargs : dict
                Additional keyword arguments for the parent `manifold_compression` class.
        """
        manifold_function = spectral_embedding_manifold(n_components=D)
        super().__init__(D=D, name=name, manifold_function=manifold_function, *args, **kwargs)


class tsne_manifold_compression(manifold_compression):
    """
        A manifold-based dimensionality reduction class using t-SNE (t-distributed Stochastic Neighbor Embedding).

        This class applies the t-SNE algorithm to reduce the dimensionality of input data, maintaining
        the local structure of the data in the reduced space.

        Methods
        -------
        __init__(D, perplexity, name='tsne_manifold_compression', *args, **kwargs)
            Initializes the t-SNE-based dimensionality reduction instance.

        Parameters
        ----------
        D : int
            Number of dimensions to retain after compression.
        perplexity : float
            Perplexity parameter for the t-SNE algorithm, which controls the balance between local and global
            aspects of the data.
        name : str, optional
            Name of the transformation. Defaults to 'tsne_manifold_compression'.
        *args : tuple
            Additional positional arguments for the parent `manifold_compression` class.
        **kwargs : dict
            Additional keyword arguments for the parent `manifold_compression` class.
    """
    def __init__(self, D: int, perplexity: float, name='tsne_manifold_compression', *args, **kwargs):
        """
            Initializes the t-SNE-based dimensionality reduction instance.

            Parameters
            ----------
            D : int
                Number of dimensions to retain after compression.
            perplexity : float
                Perplexity parameter for the t-SNE algorithm, which controls the balance between local and global
                aspects of the data.
            name : str, optional
                Name of the transformation. Defaults to 'tsne_manifold_compression'.
            *args : tuple
                Additional positional arguments for the parent `manifold_compression` class.
            **kwargs : dict
                Additional keyword arguments for the parent `manifold_compression` class.
        """
        manifold_function = tsne_manifold(n_components=D, perplexity=perplexity)
        super().__init__(D=D, name=name, manifold_function=manifold_function, *args, **kwargs)
