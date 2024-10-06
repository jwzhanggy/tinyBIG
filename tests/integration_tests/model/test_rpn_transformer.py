# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# Test RPN Transformer #
#######################

import pytest

import torch

from tinybig.model import transformer

device = 'mps'

@pytest.fixture
def sample_data_batch():
    X = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    ).to(device)
    X2 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    ).to(device)
    y = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1]
    ).to(device)
    return X, X2, y

def test_graph_interdependence(sample_data_batch):
    X, X2, y = sample_data_batch

    layer = transformer(
        dims=[3, 2],
        fc_dims=[128, 10],
        with_residual=True,
        r_interdependence=5,
        device=device
    )

    print(X.shape, layer(x=X, device=device).shape)
    print(X2.shape, layer(x=X2, device=device).shape)



