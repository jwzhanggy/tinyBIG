# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Test Geometric Interdependence #
##################################

import pytest
import torch
from tinybig.head.grid_based_heads import conv_head, pooling_head
from tinybig.interdependence import geometric_interdependence

device = 'mps'

@pytest.mark.parametrize("patch_type", ['cuboid', 'cylinder', 'sphere'])
def test_conv_head_initialization(patch_type):
    """
    Test the initialization of geometric_interdependence class with different patches.
    """
    x = torch.tensor([list(range(1200))] * 5, dtype=torch.float32, device=device)

    head = conv_head(
        h=10, w=8, d=5, in_channel=3, out_channel=10,
        patch_shape=patch_type,
        p_h=1, p_w=1, p_d=1, p_r=3,
        packing_strategy='sparse_square',
        device=device
    )

    print(head.get_patch_size())
    print(head.get_input_grid_shape())
    print(head.get_output_grid_shape())
    print(x.shape, head(x=x, device=device).shape)


@pytest.mark.parametrize("patch_type", ['cuboid', 'cylinder', 'sphere'])
def test_pooling_head_initialization(patch_type):
    """
    Test the initialization of geometric_interdependence class with different patches.
    """
    x = torch.tensor([list(range(1200))] * 5, dtype=torch.float32, device=device)

    head = pooling_head(
        h=10, w=8, d=5, channel_num=3,
        patch_shape=patch_type,
        p_h=1, p_w=1, p_d=1, p_r=3,
        packing_strategy='sparse_square',
        device=device
    )

    print(head.get_patch_size())
    print(head.get_input_grid_shape())
    print(head.get_output_grid_shape())
    print(x.shape, head(x=x, device=device).shape)



