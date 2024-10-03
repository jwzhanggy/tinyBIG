# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################################
# Test Metric based Compression Function #
##########################################

from typing import Callable

import pytest
import torch
import numpy as np
from tinybig.koala.statistics import batch_mean, batch_median
from tinybig.koala.linear_algebra import batch_sum, batch_max, batch_min
from tinybig.compression.metric_based_compression import metric_compression, max_compression, min_compression, sum_compression, mean_compression, prod_compression, median_compression  # Adjust the import according to your file structure

# Mock implementations of the metric functions for testing purposes
def mock_batch_mean(x):
    return batch_mean(x)

def mock_batch_max(x):
    return batch_max(x)

def mock_batch_min(x):
    return batch_min(x)

def mock_batch_sum(x):
    return batch_sum(x)

def mock_batch_median(x):
    return batch_median(x)

@pytest.fixture
def setup_metric_compression():
    return metric_compression(metric=mock_batch_mean)  # Use mock mean

def test_metric_based_initialization(setup_metric_compression):
    metric_comp = setup_metric_compression
    assert metric_comp.metric is not None

def test_max_compression_initialization():
    max_comp = max_compression()
    assert isinstance(max_comp.metric, Callable)

def test_min_compression_initialization():
    min_comp = min_compression()
    assert isinstance(min_comp.metric, Callable)

def test_sum_compression_initialization():
    sum_comp = sum_compression()
    assert isinstance(sum_comp.metric, Callable)

def test_mean_compression_initialization():
    mean_comp = mean_compression()
    assert isinstance(mean_comp.metric, Callable)

def test_prod_compression_initialization():
    prod_comp = prod_compression()
    assert isinstance(prod_comp.metric, Callable)

def test_median_compression_initialization():
    median_comp = median_compression()
    assert isinstance(median_comp.metric, Callable)

def test_metric_based_forward(setup_metric_compression):
    metric_comp = setup_metric_compression
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = metric_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, 1)

def test_max_compression_forward():
    max_comp = max_compression()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = max_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, 1)

def test_min_compression_forward():
    min_comp = min_compression()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = min_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, 1)

def test_sum_compression_forward():
    sum_comp = sum_compression()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = sum_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, 1)

def test_mean_compression_forward():
    mean_comp = mean_compression()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = mean_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, 1)

def test_median_compression_forward():
    median_comp = median_compression()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = median_comp.forward(x)
    assert result.shape == (2, 1)  # Expecting (batch_size, 1)

# Run the tests
if __name__ == "__main__":
    pytest.main()
