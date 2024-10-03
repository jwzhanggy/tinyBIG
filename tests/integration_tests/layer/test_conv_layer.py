# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Test Geometric Interdependence #
##################################

import pytest
import torch

from tinybig.layer.grid_based_layers import conv_layer, pooling_layer

device = 'mps'

@pytest.mark.parametrize("patch_type", ['cuboid', 'cylinder', 'sphere'])
def test_conv_layer_initialization(patch_type):
    """
    Test the initialization of geometric_interdependence class with different patches.
    """
    x = torch.tensor([list(range(1200))] * 5, dtype=torch.float32, device=device)

    print('test conv layer')

    layer_1 = conv_layer(
        h=10, w=8, d=5, in_channel=3, out_channel=10,
        width=2,
        patch_shape=patch_type,
        p_h=1, p_w=1, p_d=1, p_r=3,
        packing_strategy='dentest_packing',
        device=device,
    )

    layer_2 = conv_layer(
        h=10, w=8, d=5, in_channel=10, out_channel=20,
        width=2,
        patch_shape=patch_type,
        p_h=1, p_w=1, p_d=1, p_r=3,
        packing_strategy='dentest_packing',
        device=device,
    )

    print('input x', x.shape)
    x = layer_1(x, device=device)
    print('mid x', x.shape)
    x = layer_2(x, device=device)
    print('output x', x.shape)


@pytest.mark.parametrize("patch_type", ['cuboid', 'cylinder', 'sphere'])
def test_pooling_layer_initialization(patch_type):
    """
    Test the initialization of geometric_interdependence class with different patches.
    """

    print('test conv layer')

    x = torch.tensor([list(range(1200))] * 5, dtype=torch.float32, device=device)

    layer = pooling_layer(
        h=10, w=8, d=5, channel_num=3,
        patch_shape=patch_type,
        p_h=1, p_w=1, p_d=1, p_r=3,
        packing_strategy='dentest_packing',
        device=device
    )

    print(patch_type, "pooling: m, n", layer.get_m(), layer.get_n())
    print(patch_type, "pooling: x, kappa_x", x.shape, layer(x=x, device=device).shape)



