# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Test Base Fusion Class #
##########################

import pytest
import torch
from tinybig.module.base_transformation import transformation  # Adjust the import according to your file structure

# A mock implementation of a derived class for testing
class TestTransformation(transformation):
    def calculate_D(self, m: int) -> int:
        return m * 2  # Example expansion to double the input dimension

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs) -> torch.Tensor:
        return x * 2  # Example operation: double the input tensor

# Test cases
@pytest.fixture
def setup_transformation():
    return TestTransformation(preprocess_functions=None, postprocess_functions=None)

def test_initialization(setup_transformation):
    trans = setup_transformation
    assert trans.get_name() == 'base_transformation'
    assert trans.preprocess_functions is None
    assert trans.postprocess_functions is None
    assert trans.device == 'cpu'

def test_calculate_D(setup_transformation):
    trans = setup_transformation
    assert trans.calculate_D(3) == 6
    assert trans.calculate_D(5) == 10

def test_pre_process_no_functions(setup_transformation):
    trans = setup_transformation
    x = torch.tensor([1.0, 2.0, 3.0])
    processed = trans.pre_process(x)
    assert torch.equal(processed, x)  # No preprocessing should return the same tensor

def test_post_process_no_functions(setup_transformation):
    trans = setup_transformation
    x = torch.tensor([1.0, 2.0, 3.0])
    processed = trans.post_process(x)
    assert torch.equal(processed, x)  # No postprocessing should return the same tensor

def test_forward(setup_transformation):
    trans = setup_transformation
    x = torch.tensor([1.0, 2.0, 3.0])
    result = trans.forward(x)
    assert torch.equal(result, torch.tensor([2.0, 4.0, 6.0]))  # Expecting doubled values

def test_call_method(setup_transformation):
    trans = setup_transformation
    x = torch.tensor([1.0, 2.0, 3.0])
    result = trans(x)
    assert torch.equal(result, torch.tensor([2.0, 4.0, 6.0]))  # Expecting doubled values

def test_to_config(setup_transformation):
    trans = setup_transformation
    config = trans.to_config()
    print(config)
    assert config['function_class'].endswith("TestTransformation")
    assert config['function_parameters']['name'] == 'base_transformation'

# Run the tests
if __name__ == "__main__":
    pytest.main()
