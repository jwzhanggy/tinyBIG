# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################################################
# Test Incremental Dimension Reduction in koala.machine_learning #
##################################################################

import pytest
import numpy as np
import torch

from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import SparseRandomProjection
from tinybig.koala.machine_learning.dimension_reduction import incremental_dimension_reduction, incremental_PCA, incremental_random_projection


def generate_numpy_data(n_samples=100, n_features=50):
    return np.random.rand(n_samples, n_features)


def generate_torch_data(n_samples=100, n_features=50):
    return torch.rand(n_samples, n_features)


class Test_Incremental_Dimension_Reduction:

    def test_initialization(self):
        dr = incremental_dimension_reduction(n_feature=10)
        assert dr.name == 'incremental_dimension_reduction'
        assert dr.n_feature == 10
        assert dr.incremental is True

    def test_set_n_feature(self):
        dr = incremental_dimension_reduction()
        dr.set_n_feature(15)
        assert dr.get_n_feature() == 15


class Test_Incremental_PCA:

    def test_initialization(self):
        ipca = incremental_PCA(n_feature=10)
        assert ipca.name == 'incremental_PCA'
        assert ipca.n_feature == 10
        assert isinstance(ipca.ipca, IncrementalPCA)

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    def test_fit_transform(self, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        ipca = incremental_PCA(n_feature=10)
        X_reduced = ipca.fit_transform(X)

        if input_type == torch.Tensor:
            assert isinstance(X_reduced, torch.Tensor)
        else:
            assert isinstance(X_reduced, np.ndarray)

        assert X_reduced.shape == (100, 10)

    def test_update_n_feature(self):
        X = generate_numpy_data(100, 50)
        ipca = incremental_PCA(n_feature=10)
        ipca.fit(X)

        ipca.update_n_feature(20)
        assert ipca.get_n_feature() == 20

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    def test_transform_assertions(self, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        ipca = incremental_PCA(n_feature=10)
        ipca.fit(X)
        X_reduced = ipca.transform(X)

        assert X_reduced.shape[1] == 10
        with pytest.raises(AssertionError):
            ipca = incremental_PCA(n_feature=60)
            ipca.transform(X)


class Test_Incremental_Random_Projection:

    def test_initialization(self):
        irp = incremental_random_projection(n_feature=10)
        assert irp.name == 'incremental_random_projection'
        assert irp.n_feature == 10
        assert isinstance(irp.irp, SparseRandomProjection)

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    def test_fit_transform(self, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        irp = incremental_random_projection(n_feature=10)
        X_reduced = irp.fit_transform(X)

        print(X.dtype, X_reduced.dtype)

        if input_type == torch.Tensor:
            assert isinstance(X_reduced, torch.Tensor)
        else:
            assert isinstance(X_reduced, np.ndarray)

        assert X_reduced.shape == (100, 10)

    def test_update_n_feature(self):
        X = generate_numpy_data(100, 50)
        irp = incremental_random_projection(n_feature=10)
        irp.fit(X)

        irp.update_n_feature(20)
        assert irp.get_n_feature() == 20

    @pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
    def test_transform_assertions(self, input_type):
        if input_type == np.ndarray:
            X = generate_numpy_data(100, 50)
        else:
            X = generate_torch_data(100, 50)

        irp = incremental_random_projection(n_feature=10)
        irp.fit(X)
        X_reduced = irp.transform(X)

        assert X_reduced.shape[1] == 10
        with pytest.raises(AssertionError):
            irp = incremental_random_projection(n_feature=60)
            irp.transform(X)


@pytest.mark.parametrize("n_feature", [5, 10, 20])
def test_parametrized_fit_transform(n_feature):
    X = generate_numpy_data(100, 50)
    ipca = incremental_PCA(n_feature=n_feature)
    X_reduced = ipca.fit_transform(X)
    assert X_reduced.shape == (100, n_feature)

    irp = incremental_random_projection(n_feature=n_feature)
    X_reduced_rp = irp.fit_transform(X)
    assert X_reduced_rp.shape == (100, n_feature)
