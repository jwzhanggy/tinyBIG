# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################
# Other Utility Functions #
###########################

import torch
import random
import numpy as np
import os

import importlib
import pkgutil
import inspect
import requests


def set_random_seed(random_seed: int = 0):
    """
    Random seed setup method.

    It sets up the random seeds for the RPN model prior to model training and testing.

    Specifically, this method will set up the random seeds and related configurations of multiple packages,
    including
    * numpy
    * random
    * torch
    * torch.cuda
    * torch.cudnn
    * torch.mps

    Parameters
    ----------
    random_seed: int, default = 0
        The random seed to be setup.

    Returns
    -------
    None
        This method doesn't have any return values.
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(random_seed)


def check_directory_exists(complete_file_path):
    directory_path = os.path.dirname(complete_file_path)
    return os.path.exists(directory_path)


def create_directory_if_not_exists(complete_file_path):
    """
    The directory creation method.

    It checks whether the target file directory exists or not,
    if it doesn't exist, this method will create the directory.

    Parameters
    ----------
    complete_file_path: str
        The complete file path (covering the directory and file name) as a string.

    Returns
    -------
    None
        This method doesn't have any return values.
    """
    directory_path = os.path.dirname(complete_file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' doesn't exit, and it was created...")


def async_clear_tensor_memory(tensor):
    if tensor is None:
        print("Tensor is None, nothing to clear.")
        return

    device = tensor.device

    if device.type == 'cuda':
        # Create a CUDA stream
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Move tensor to CPU asynchronously
            tensor_cpu = tensor.cpu()
            del tensor
            # Clear unused GPU memory
            torch.cuda.empty_cache()
            print("Memory cleared for CUDA tensor.")
    elif device.type == 'mps':
        # MPS specific handling (if applicable)
        tensor_cpu = tensor.cpu()
        del tensor
        print("Memory cleared for MPS tensor.")
    else:
        del tensor
        print(f"Memory cleared for tensor on {device.type} device.")

    # Force garbage collection
    import gc
    gc.collect()


def find_class_in_package(class_name: str, package_name: str = 'tinybig'):
    try:
        # Import the package
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        return f"Package '{package_name}' not found"

    # List all modules in the package
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        if not is_pkg:  # We are only interested in modules, not sub-packages
            try:
                # Dynamically import the module
                module = importlib.import_module(module_name)

                # Check if the class exists in the module
                if hasattr(module, class_name):
                    # Ensure that it is indeed a class and not a different attribute
                    class_obj = getattr(module, class_name)
                    if inspect.isclass(class_obj):
                        # Return the full class path
                        return f"{module_name}.{class_name}"
            except ImportError:
                continue

    return f"Class '{class_name}' not found in package '{package_name}'"


def download_file_from_github(url_link: str, destination_path: str):
    create_directory_if_not_exists(destination_path)

    response = requests.get(url_link, stream=True)
    if response.status_code == 200:
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(1024):  # Download in chunks
                file.write(chunk)
        print(f"File downloaded successfully: {destination_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


if __name__ == '__main__':
    # Example usage:
    url = "https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/graph/cora/node"  # Replace with your file's raw URL
    destination = "./data/graph/cora/node"  # Specify where to save the file
    download_file_from_github(url, destination)
