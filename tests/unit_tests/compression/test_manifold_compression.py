# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

############################################
# Test Manifold based Compression Function #
############################################


import pytest
import torch
from tinybig.koala.manifold import manifold, isomap_manifold, lle_manifold, mds_manifold, spectral_embedding_manifold, tsne_manifold
from tinybig.compression.manifold_compression import manifold_compression, isomap_manifold_compression, lle_manifold_compression, mds_manifold_compression, spectral_embedding_manifold_compression, tsne_manifold_compression  # Adjust the import according to your file structure

# Mock implementations of the manifold functions for testing purposes
class MockIsomapManifold(isomap_manifold):
    def __call__(self, x: torch.Tensor):
        return x[:, :self.n_components]  # Just returns the first n_components dimensions

class MockLLEManifold(lle_manifold):
    def __call__(self, x: torch.Tensor):
        return x[:, :self.n_components]  # Just returns the first n_components dimensions

class MockMDSManifold(mds_manifold):
    def __call__(self, x: torch.Tensor):
        return x[:, :self.n_components]  # Just returns the first n_components dimensions

class MockSpectralEmbeddingManifold(spectral_embedding_manifold):
    def __call__(self, x: torch.Tensor):
        return x[:, :self.n_components]  # Just returns the first n_components dimensions

class MockTSNEManifold(tsne_manifold):
    def __call__(self, x: torch.Tensor):
        return x[:, :self.n_components]  # Just returns the first n_components dimensions

@pytest.fixture
def setup_manifold_compression():
    manifold_function = MockIsomapManifold(n_components=1)  # Mock Isomap with 2 components
    return manifold_compression(D=1, manifold_function=manifold_function)

@pytest.fixture
def setup_isomap_manifold_compression():
    return isomap_manifold_compression(D=1)  # 2 components

@pytest.fixture
def setup_lle_manifold_compression():
    return lle_manifold_compression(D=1)  # 2 components

@pytest.fixture
def setup_mds_manifold_compression():
    return mds_manifold_compression(D=1)  # 3 components

@pytest.fixture
def setup_spectral_embedding_manifold_compression():
    return spectral_embedding_manifold_compression(D=1)  # 3 components

@pytest.fixture
def setup_tsne_manifold_compression():
    return tsne_manifold_compression(D=1, perplexity=1.0)  # 3 components

def test_manifold_initialization(setup_manifold_compression):
    manifold_comp = setup_manifold_compression
    assert manifold_comp.D == 1
    assert isinstance(manifold_comp.manifold_function, MockIsomapManifold)

def test_isomap_manifold_compression_initialization(setup_isomap_manifold_compression):
    isomap_comp = setup_isomap_manifold_compression
    assert isomap_comp.D == 1
    assert isinstance(isomap_comp.manifold_function, isomap_manifold)

def test_lle_manifold_compression_initialization(setup_lle_manifold_compression):
    lle_comp = setup_lle_manifold_compression
    assert lle_comp.D == 1
    assert isinstance(lle_comp.manifold_function, lle_manifold)

def test_mds_manifold_compression_initialization(setup_mds_manifold_compression):
    mds_comp = setup_mds_manifold_compression
    assert mds_comp.D == 1
    assert isinstance(mds_comp.manifold_function, mds_manifold)

def test_spectral_embedding_manifold_compression_initialization(setup_spectral_embedding_manifold_compression):
    spectral_comp = setup_spectral_embedding_manifold_compression
    assert spectral_comp.D == 1
    assert isinstance(spectral_comp.manifold_function, spectral_embedding_manifold)

def test_tsne_manifold_compression_initialization(setup_tsne_manifold_compression):
    tsne_comp = setup_tsne_manifold_compression
    assert tsne_comp.D == 1
    assert isinstance(tsne_comp.manifold_function, tsne_manifold)

def test_manifold_forward(setup_manifold_compression):
    manifold_comp = setup_manifold_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = manifold_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 2)

def test_isomap_manifold_compression_forward(setup_isomap_manifold_compression):
    isomap_comp = setup_isomap_manifold_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = isomap_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 2)

def test_lle_manifold_compression_forward(setup_lle_manifold_compression):
    lle_comp = setup_lle_manifold_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = lle_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 2)

def test_mds_manifold_compression_forward(setup_mds_manifold_compression):
    mds_comp = setup_mds_manifold_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = mds_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 3)

def test_spectral_embedding_manifold_compression_forward(setup_spectral_embedding_manifold_compression):
    spectral_comp = setup_spectral_embedding_manifold_compression
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]])
    result = spectral_comp.forward(x)
    assert result.shape == (3, 1)  # Expecting (batch_size, D) -> (3, 1)

def test_tsne_manifold_compression_forward(setup_tsne_manifold_compression):
    tsne_comp = setup_tsne_manifold_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = tsne_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, D) -> (2, 3)

# Run the tests
if __name__ == "__main__":
    pytest.main()
