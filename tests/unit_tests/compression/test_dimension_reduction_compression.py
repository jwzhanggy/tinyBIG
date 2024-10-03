# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################################################
# Test Dimension Reduction based Compression Function #
#######################################################

import pytest
import torch
from tinybig.koala.machine_learning.dimension_reduction import incremental_dimension_reduction, incremental_PCA, incremental_random_projection
from tinybig.compression.dimension_reduction_compression import dimension_reduction_compression, incremental_PCA_based_compression, incremental_random_projection_based_compression  # Adjust the import according to your file structure


# Mock implementations of the dimension reduction functions for testing purposes
class MockIncrementalPCA(incremental_PCA):
    def __call__(self, x: torch.Tensor):
        return x[:, :self.n_feature]  # Just returns the first n_feature dimensions

class MockIncrementalRandomProjection(incremental_random_projection):
    def __call__(self, x: torch.Tensor):
        return x[:, :self.n_feature]  # Just returns the first n_feature dimensions

@pytest.fixture
def setup_dimension_reduction_compression():
    dr_function = MockIncrementalPCA(n_feature=1)  # Mock PCA with 2 features
    return dimension_reduction_compression(D=1, dr_function=dr_function)

@pytest.fixture
def setup_incremental_PCA_based_compression():
    return incremental_PCA_based_compression(D=1)  # 2 features

@pytest.fixture
def setup_incremental_random_projection_based_compression():
    return incremental_random_projection_based_compression(D=1)  # 3 features

def test_dimension_reduction_initialization(setup_dimension_reduction_compression):
    dr_comp = setup_dimension_reduction_compression
    assert dr_comp.D == 1
    assert isinstance(dr_comp.dr_function, MockIncrementalPCA)

def test_incremental_PCA_based_compression_initialization(setup_incremental_PCA_based_compression):
    pca_comp = setup_incremental_PCA_based_compression
    assert pca_comp.D == 1
    assert isinstance(pca_comp.dr_function, incremental_PCA)

def test_incremental_random_projection_based_compression_initialization(setup_incremental_random_projection_based_compression):
    rp_comp = setup_incremental_random_projection_based_compression
    assert rp_comp.D == 1
    assert isinstance(rp_comp.dr_function, incremental_random_projection)

def test_dimension_reduction_forward(setup_dimension_reduction_compression):
    dr_comp = setup_dimension_reduction_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = dr_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 2)

def test_incremental_PCA_based_compression_forward(setup_incremental_PCA_based_compression):
    pca_comp = setup_incremental_PCA_based_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = pca_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 2)

def test_incremental_random_projection_based_compression_forward(setup_incremental_random_projection_based_compression):
    rp_comp = setup_incremental_random_projection_based_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = rp_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 3)

# Run the tests
if __name__ == "__main__":
    pytest.main()

