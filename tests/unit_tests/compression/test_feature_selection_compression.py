# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################################
# Test Feature Selection based Compression Function #
#####################################################


import pytest
import torch
from tinybig.koala.machine_learning.feature_selection import feature_selection, incremental_feature_clustering, incremental_variance_threshold
from tinybig.compression.feature_selection_compression import feature_selection_compression, incremental_feature_clustering_based_compression, incremental_variance_threshold_based_compression  # Adjust the import according to your file structure

# Mock implementations of the feature selection functions for testing purposes
class MockIncrementalFeatureClustering(incremental_feature_clustering):
    def __call__(self, x: torch.Tensor):
        return x[:, :self.n_feature]  # Just returns the first n_feature dimensions

class MockIncrementalVarianceThreshold(incremental_variance_threshold):
    def __call__(self, x: torch.Tensor):
        return x[:, :self.n_feature]  # Just returns the first n_feature dimensions

@pytest.fixture
def setup_feature_selection_compression():
    fs_function = MockIncrementalFeatureClustering(n_feature=1)  # Mock clustering with 2 features
    return feature_selection_compression(D=1, fs_function=fs_function)

@pytest.fixture
def setup_incremental_feature_clustering_based_compression():
    return incremental_feature_clustering_based_compression(D=1)  # 2 features

@pytest.fixture
def setup_incremental_variance_threshold_based_compression():
    return incremental_variance_threshold_based_compression(D=1)  # 3 features

def test_feature_selection_initialization(setup_feature_selection_compression):
    fs_comp = setup_feature_selection_compression
    assert fs_comp.D == 1
    assert isinstance(fs_comp.fs_function, MockIncrementalFeatureClustering)

def test_incremental_feature_clustering_based_compression_initialization(setup_incremental_feature_clustering_based_compression):
    clustering_comp = setup_incremental_feature_clustering_based_compression
    assert clustering_comp.D == 1
    assert isinstance(clustering_comp.fs_function, incremental_feature_clustering)

def test_incremental_variance_threshold_based_compression_initialization(setup_incremental_variance_threshold_based_compression):
    variance_comp = setup_incremental_variance_threshold_based_compression
    assert variance_comp.D == 1
    assert isinstance(variance_comp.fs_function, incremental_variance_threshold)

def test_feature_selection_forward(setup_feature_selection_compression):
    fs_comp = setup_feature_selection_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = fs_comp.forward(x)
    print(result)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 2)

def test_incremental_feature_clustering_based_compression_forward(setup_incremental_feature_clustering_based_compression):
    clustering_comp = setup_incremental_feature_clustering_based_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = clustering_comp.forward(x)
    print(result)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 2)

def test_incremental_variance_threshold_based_compression_forward(setup_incremental_variance_threshold_based_compression):
    variance_comp = setup_incremental_variance_threshold_based_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = variance_comp.forward(x)
    print(result)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 3)

# Run the tests
if __name__ == "__main__":
    pytest.main()
