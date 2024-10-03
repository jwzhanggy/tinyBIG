# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Test Geometric Interdependence #
##################################

import pytest
import torch

from tinybig.model.rpn_cnn import cnn

device = 'mps'


@pytest.mark.parametrize("patch_type", ['cuboid', 'cylinder', 'sphere'])
def test_conv_layer_initialization(patch_type):
    """
    Test the initialization of geometric_interdependence class with different patches.
    """
    x = torch.tensor([list(range(1200))] * 5, dtype=torch.float32, device=device)

    model = cnn(
        h=10, w=8, d=5,
        channel_nums=[3, 128, 256, 256, 512],
        fc_dims=[512, 128, 10], width=2,
        patch_shape=patch_type, p_h=1, p_w=1, p_d=1, p_r=3,
        cd_h=1, cd_w=1, cd_d=1,
        with_dual_lphm=False, r=3,
        device=device
    )

    print('parameter num: ', sum([parameter.numel() for parameter in model.parameters()]))

    x = model(x, device=device)
    assert x.shape == (5, 10)
