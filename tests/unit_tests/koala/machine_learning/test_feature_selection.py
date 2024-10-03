# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################################################
# Test Incremental Feature Selection in koala.machine_learning #
################################################################

import pytest
import numpy as np
import torch
from tinybig.koala.machine_learning.feature_selection import feature_selection, incremental_feature_clustering, incremental_variance_threshold


# Utility functions to generate random data
def generate_numpy_data(n_samples=100, n_features=50):
    return np.random.rand(n_samples, n_features)


def generate_torch_data(n_samples=100, n_features=50):
    return torch.rand(n_samples, n_features)


# Test feature_selection base class
class Test_Feature_Selection:

    def test_initialization(self):
        fs = feature_selection(n_feature=10)
        assert fs.name == 'feature_selection'
        assert fs.n_feature == 10
        assert fs.incremental is True
        assert fs.incremental_stop_threshold == 0.01
        assert fs.t_threshold == 100

    def test_set_get_n_feature(self):
        fs = feature_selection()
        fs.set_n_feature(15)
        assert fs.get_n_feature() == 15


# Test incremental_feature_clustering
class Test_Incremental_Feature_Clustering:

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    def test_fit_transform(self, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        fc = incremental_feature_clustering(n_feature=10)
        X_reduced = fc.fit_transform(X)

        if input_type == torch.Tensor:
            assert isinstance(X_reduced, torch.Tensor)
        else:
            assert isinstance(X_reduced, np.ndarray)

        assert X_reduced.shape == (100, 10)  # Shape should match n_feature x input feature size

    def test_update_n_feature(self):
        X = generate_numpy_data(100, 50)
        fc = incremental_feature_clustering(n_feature=10)
        fc.fit(X)

        fc.update_n_feature(20)
        assert fc.get_n_feature() == 20

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    def test_transform_assertions(self, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        fc = incremental_feature_clustering(n_feature=10)
        fc.fit(X)
        X_reduced = fc.transform(X)

        assert X_reduced.shape[1] == 10
        with pytest.raises(AssertionError):
            fc = incremental_feature_clustering(n_feature=60)
            fc.transform(X)


# Test incremental_variance_threshold
class Test_Incremental_Variance_Threshold:

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    def test_fit_transform(self, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        ivt = incremental_variance_threshold(threshold=0.1, n_feature=10)
        X_reduced = ivt.fit_transform(X)

        if input_type == torch.Tensor:
            assert isinstance(X_reduced, torch.Tensor)
        else:
            assert isinstance(X_reduced, np.ndarray)

        assert X_reduced.shape == (100, 10)  # Shape should match input samples x n_feature

    def test_update_threshold(self):
        X = generate_numpy_data(100, 50)
        ivt = incremental_variance_threshold(threshold=0.1, n_feature=10)
        ivt.fit(X)

        ivt.update_threshold(0.2)
        assert ivt.threshold == 0.2

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    def test_transform_assertions(self, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        ivt = incremental_variance_threshold(threshold=0.1, n_feature=10)
        ivt.fit(X)
        X_reduced = ivt.transform(X)

        assert X_reduced.shape[1] == 10
        with pytest.raises(AssertionError):
            ivt = incremental_variance_threshold(n_feature=60)
            ivt.transform(X)


@pytest.mark.parametrize("n_feature", [5, 10, 20])
def test_parametrized_fit_transform(n_feature):
    X = generate_numpy_data(100, 50)

    # Test incremental_feature_clustering
    fc = incremental_feature_clustering(n_feature=n_feature)
    X_reduced = fc.fit_transform(X)
    assert X_reduced.shape == (100, n_feature)

    # Test incremental_variance_threshold
    ivt = incremental_variance_threshold(n_feature=n_feature)
    X_reduced_vt = ivt.fit_transform(X)
    assert X_reduced_vt.shape == (100, n_feature)

