# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################################################
# Test Incremental Dimension Reduction in koala.machine_learning #
##################################################################

import pytest
import numpy as np
import torch
from tinybig.koala.manifold import isomap_manifold, lle_manifold, mds_manifold, spectral_embedding_manifold, tsne_manifold


# Utility functions to generate random data in both numpy and torch formats
def generate_numpy_data(n_samples=100, n_features=50):
    return np.random.rand(n_samples, n_features)


def generate_torch_data(n_samples=100, n_features=50):
    return torch.rand(n_samples, n_features)


# Test for manifold subclasses
class TestManifoldSubclasses:

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    @pytest.mark.parametrize("manifold_class", [isomap_manifold, lle_manifold, mds_manifold, spectral_embedding_manifold, tsne_manifold])
    def test_fit_transform(self, manifold_class, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        m = manifold_class(n_neighbors=10, n_components=3)
        X_transformed = m.fit_transform(X)

        # Ensure correct data type is returned
        if isinstance(X, torch.Tensor):
            assert isinstance(X_transformed, torch.Tensor), "Output should be torch when input is torch"
        else:
            assert isinstance(X_transformed, np.ndarray), "Output should be numpy for numpy input"

        assert X_transformed.shape == (100, 3)  # Check the reduced dimensions

    def test_init_model(self):
        m = isomap_manifold(n_neighbors=5, n_components=2)
        old_model = m.model
        m.set_n_neighbors(15)
        assert m.get_n_neighbors() == 15
        m.init_model()
        assert m.model != old_model

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    @pytest.mark.parametrize("manifold_class", [isomap_manifold, lle_manifold, mds_manifold, spectral_embedding_manifold, tsne_manifold])
    def test_transform(self, manifold_class, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        m = manifold_class(n_neighbors=5, n_components=2)
        X_transformed = m.fit_transform(X)  # Apply transform

        assert X_transformed.shape == (100, 2)
        if isinstance(X, torch.Tensor):
            assert isinstance(X_transformed, torch.Tensor), "Output should be torch when input is torch"
        else:
            assert isinstance(X_transformed, np.ndarray), "Output should be numpy for numpy input"

@pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
@pytest.mark.parametrize("manifold_class", [isomap_manifold, lle_manifold, mds_manifold, spectral_embedding_manifold, tsne_manifold])
def test_parametrized_manifold_fit_transform(manifold_class, input_type):
    if input_type == np.ndarray:
        X = generate_numpy_data(200, 50)
    else:
        X = generate_torch_data(200, 50)

    m = manifold_class(n_neighbors=5, n_components=2)
    X_transformed = m.fit_transform(X)

    assert X_transformed.shape == (200, 2)
    if isinstance(X, torch.Tensor):
        assert isinstance(X_transformed, torch.Tensor), "Output should be torch when input is torch"
    else:
        assert isinstance(X_transformed, np.ndarray), "Output should be numpy for numpy input"

