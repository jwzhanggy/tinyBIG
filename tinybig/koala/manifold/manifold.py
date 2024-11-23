# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############
# Manifolds #
#############

from typing import Union

import torch
import numpy as np

from abc import abstractmethod

from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS, SpectralEmbedding, smacof


class manifold(object):
    """
        Base class for manifold learning.

        This class provides a generic interface for manifold learning algorithms, with methods to set the number of
        neighbors and components and to perform dimensionality reduction using a specified manifold learning technique.

        Attributes
        ----------
        name : str
            The name of the manifold learning method.
        n_neighbors : int
            The number of neighbors to consider for the manifold learning algorithm.
        n_components : int
            The number of components for dimensionality reduction.
        model : object
            The manifold learning model to be used.

        Methods
        -------
        get_n_neighbors()
            Get the number of neighbors used in the manifold learning algorithm.
        set_n_neighbors(n_neighbors)
            Set the number of neighbors and reinitialize the model.
        get_n_components()
            Get the number of components for dimensionality reduction.
        set_n_components(n_components)
            Set the number of components and reinitialize the model.
        forward(X, device='cup', *args, **kwargs)
            Apply the manifold learning algorithm to reduce the dimensionality of the input data.
        fit_transform(X, device='cup', *args, **kwargs)
            Perform dimensionality reduction on the input data.
        init_model()
            Abstract method to initialize the manifold learning model.
    """
    def __init__(self, name: str = 'base_manifold', n_neighbors: int = 5, n_components: int = 2, *args, **kwargs):
        """
            Initialize the manifold learning class.

            Parameters
            ----------
            name : str, optional
                The name of the manifold learning method. Default is 'base_manifold'.
            n_neighbors : int, optional
                The number of neighbors to consider. Default is 5.
            n_components : int, optional
                The number of components for dimensionality reduction. Default is 2.
            *args, **kwargs
                Additional arguments for initialization.
        """
        self.name = name
        self.n_neighbors = n_neighbors
        self.n_components = n_components

        self.model = None
        self.init_model()

    def get_n_neighbors(self):
        """
            Get the number of neighbors used in the manifold learning algorithm.

            Returns
            -------
            int
                The number of neighbors.
        """
        return self.n_neighbors

    def set_n_neighbors(self, n_neighbors):
        """
            Set the number of neighbors and reinitialize the model.

            Parameters
            ----------
            n_neighbors : int
                The new number of neighbors.
        """
        self.n_neighbors = n_neighbors
        self.init_model()

    def get_n_components(self):
        """
        Get the number of components for dimensionality reduction.

        Returns
        -------
        int
            The number of components.
        """
        return self.n_components

    def set_n_components(self, n_components):
        """
        Set the number of components and reinitialize the model.

        Parameters
        ----------
        n_components : int
            The new number of components.
        """
        self.n_components = n_components
        self.init_model()

    def __call__(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
        Apply the manifold learning algorithm to the input data.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The input data for dimensionality reduction.
        device : str, optional
            The device to use ('cup' or 'cpu'). Default is 'cup'.
        *args, **kwargs
            Additional arguments.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The transformed data.
        """
        return self.forward(X=X, device=device, *args, **kwargs)

    def forward(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
        Perform dimensionality reduction on the input data.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The input data for dimensionality reduction.
        device : str, optional
            The device to use ('cup' or 'cpu'). Default is 'cup'.
        *args, **kwargs
            Additional arguments.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The transformed data.
        """
        return self.fit_transform(X=X, device=device, *args, **kwargs)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cup', *args, **kwargs):
        """
        Perform dimensionality reduction on the input data.

        Parameters
        ----------
        X : Union[np.ndarray, torch.Tensor]
            The input data for dimensionality reduction.
        device : str, optional
            The device to use ('cup' or 'cpu'). Default is 'cup'.
        *args, **kwargs
            Additional arguments.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            The transformed data.
        """
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()  # Convert torch.Tensor to numpy
        else:
            input_X = X

        X_manifold = self.model.fit_transform(X=input_X)

        return torch.tensor(X_manifold) if isinstance(X, torch.Tensor) and not isinstance(X_manifold, torch.Tensor) else X_manifold

    @abstractmethod
    def init_model(self):
        """
        Abstract method to initialize the manifold learning model.
        """
        pass


class isomap_manifold(manifold):
    """
    Manifold learning using the Isomap algorithm.

    This class implements dimensionality reduction using the Isomap algorithm, which preserves global geometric
    structure in the data.
    """
    def __init__(self, name: str = 'isomap_manifold', *args, **kwargs):
        """
        Initialize the Isomap manifold learning class.

        Parameters
        ----------
        name : str, optional
            The name of the manifold learning method. Default is 'isomap_manifold'.
        *args, **kwargs
            Additional arguments for the base class.
        """
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        """
        Initialize the Isomap model.
        """
        self.model = Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components)


class lle_manifold(manifold):
    """
    Manifold learning using the Locally Linear Embedding (LLE) algorithm.

    This class implements dimensionality reduction using the LLE algorithm, which preserves local geometric
    structure in the data.
    """
    def __init__(self, name: str = 'lle_manifold', *args, **kwargs):
        """
        Initialize the LLE manifold learning class.

        Parameters
        ----------
        name : str, optional
            The name of the manifold learning method. Default is 'lle_manifold'.
        *args, **kwargs
            Additional arguments for the base class.
        """
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        """
        Initialize the LLE model.
        """
        self.model = LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components)


class mds_manifold(manifold):
    """
    Manifold learning using the Multidimensional Scaling (MDS) algorithm.

    This class implements dimensionality reduction using the MDS algorithm, which minimizes the stress of the
    low-dimensional representation of the data.
    """
    def __init__(self, name: str = 'mds_manifold', *args, **kwargs):
        """
        Initialize the MDS manifold learning class.

        Parameters
        ----------
        name : str, optional
            The name of the manifold learning method. Default is 'mds_manifold'.
        *args, **kwargs
            Additional arguments for the base class.
        """
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        """
        Initialize the MDS model.
        """
        self.model = MDS(n_components=self.n_components)


class spectral_embedding_manifold(manifold):
    """
    Manifold learning using the Spectral Embedding algorithm.

    This class implements dimensionality reduction using the Spectral Embedding algorithm, which uses eigenvalue
    decomposition of the Laplacian matrix.
    """
    def __init__(self, name: str = 'spectral_embedding_manifold', *args, **kwargs):
        """
        Initialize the Spectral Embedding manifold learning class.

        Parameters
        ----------
        name : str, optional
            The name of the manifold learning method. Default is 'spectral_embedding_manifold'.
        *args, **kwargs
            Additional arguments for the base class.
        """
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        """
        Initialize the Spectral Embedding model.
        """
        self.model = SpectralEmbedding(n_components=self.n_components)


class tsne_manifold(manifold):
    """
    Manifold learning using the t-SNE algorithm.

    This class implements dimensionality reduction using the t-SNE algorithm, which is effective for visualizing
    high-dimensional data.
    """
    def __init__(self, perplexity: float = 5.0, name: str = 'tsne_manifold', *args, **kwargs):
        """
        Initialize the t-SNE manifold learning class.

        Parameters
        ----------
        perplexity : float
            The perplexity parameter for t-SNE.
        name : str, optional
            The name of the manifold learning method. Default is 'tsne_manifold'.
        *args, **kwargs
            Additional arguments for the base class.
        """
        self.perplexity = perplexity
        super().__init__(name=name, *args, **kwargs)

    def init_model(self):
        """
        Initialize the t-SNE model.
        """
        self.model = TSNE(n_components=self.n_components, perplexity=self.perplexity)


