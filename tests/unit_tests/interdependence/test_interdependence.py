# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############################
# Test Base Interdependence #
#############################

import pytest
import torch
import torch.nn.functional as F

from tinybig.module.base_interdependence import interdependence


class Test_Interdependence:

    @pytest.fixture
    def sample_interdependence(self):
        b = 5
        m = 3
        preprocess_functions = [
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(num_features=m, device='cpu')
        ]
        postprocess_functions = [
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(num_features=m, device='cpu')
        ]
        return interdependence(
            b=b, m=m, interdependence_type='attribute', device='cpu',
            preprocess_functions=preprocess_functions, postprocess_functions=postprocess_functions
        )

    def test_initialization(self, sample_interdependence):
        assert sample_interdependence.b == 5
        assert sample_interdependence.m == 3
        assert sample_interdependence.interdependence_type == 'attribute'
        assert sample_interdependence.device == 'cpu'

    def test_interdependence_type_validation(self):
        with pytest.raises(ValueError):
            interdependence(b=5, m=3, interdependence_type='invalid_type')

    def test_get_A_warning(self, sample_interdependence):
        with pytest.warns(UserWarning, match="The A matrix is None"):
            assert sample_interdependence.get_A() is None

    def test_getters(self, sample_interdependence):
        assert sample_interdependence.get_b() == 5
        assert sample_interdependence.get_m() == 3

    def test_preprocess_postprocess(self, sample_interdependence):
        x = torch.randn(5, 3)
        processed_x = sample_interdependence.pre_process(x)
        assert processed_x.shape == x.shape
        expected_preprocessed_x = F.batch_norm(F.gelu(x), running_mean=torch.zeros(sample_interdependence.get_m()), running_var=torch.ones(sample_interdependence.get_m()),
                                               training=True)
        assert torch.allclose(processed_x, expected_preprocessed_x, atol=1e-6)

        processed_x = sample_interdependence.post_process(x)
        assert processed_x.shape == x.shape
        expected_postprocessed_x = F.batch_norm(F.gelu(x), running_mean=torch.zeros(sample_interdependence.get_m()), running_var=torch.ones(sample_interdependence.get_m()),
                                                training=True)
        assert torch.allclose(processed_x, expected_postprocessed_x, atol=1e-6)

    def test_forward_attribute_interdependence(self, sample_interdependence):
        x = torch.randn(5, 3)
        w = torch.nn.Parameter(torch.randn(3, 3))
        with pytest.raises(AssertionError):  # as calculate_A is abstract and won't return a matrix
            sample_interdependence.forward(x=x, w=w)

    def test_forward_instance_interdependence(self):
        b, m = 4, 2
        interdep = interdependence(b=b, m=m, interdependence_type='instance', device='cpu')
        x = torch.randn(b, m)
        w = torch.nn.Parameter(torch.randn(b, b))
        with pytest.raises(AssertionError):  # as calculate_A is abstract and won't return a matrix
            interdep.forward(x=x, w=w)

    def test_to_config(self, sample_interdependence):
        config_dict = sample_interdependence.to_config()
        assert "function_class" in config_dict
        assert "function_parameters" in config_dict

    def test_invalid_forward(self, sample_interdependence):
        x = torch.randn(5, 3)
        with pytest.raises(ValueError):
            sample_interdependence.interdependence_type = 'invalid'
            sample_interdependence.forward(x=x)


if __name__ == "__main__":
    pytest.main([__file__])
