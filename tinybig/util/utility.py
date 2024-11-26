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
import zipfile


def set_random_seed(random_seed: int = 0, deterministic: bool = False):
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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if using multi-GPU.

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic:
        if not torch.backends.mps.is_available():
            torch.use_deterministic_algorithms(True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Specific to CUDA
        else:
            print("Warning: Deterministic algorithms disabled for MPS backend to avoid performance degradation.")


def check_file_existence(complete_file_path):
    """
    Checks if a file exists at the specified path.

    Parameters
    ----------
    complete_file_path : str
        The full file path to check.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    return os.path.exists(complete_file_path)


def check_directory_exists(complete_file_path):
    """
    Checks if the directory of a given file path exists.

    Parameters
    ----------
    complete_file_path : str
        The full file path whose directory needs to be checked.

    Returns
    -------
    bool
        True if the directory exists, False otherwise.
    """
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
    """
    Clears memory occupied by a tensor asynchronously.

    Moves a tensor to the CPU (if applicable) and clears associated memory.
    Supports tensors on CUDA and MPS devices.

    Parameters
    ----------
    tensor : torch.Tensor or None
        The tensor to clear. If None, no action is taken.

    Returns
    -------
    None
        This method does not return any values.
    """
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
    """
    Searches for a class within a package and its modules.

    Parameters
    ----------
    class_name : str
        The name of the class to search for.
    package_name : str, optional
        The name of the package to search within. Default is 'tinybig'.

    Returns
    -------
    str
        The full module path of the class if found, otherwise an error message.
    """
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
    """
    Downloads a file from a given URL and saves it to a destination path.

    Parameters
    ----------
    url_link : str
        The URL of the file to download.
    destination_path : str
        The path where the file will be saved.

    Returns
    -------
    None
        This method does not return any values.
    """
    create_directory_if_not_exists(destination_path)

    response = requests.get(url_link, stream=True)
    if response.status_code == 200:
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(1024):  # Download in chunks
                file.write(chunk)
        print(f"File downloaded successfully: {destination_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def unzip_file(complete_file_path: str, destination: str = None):
    """
    Unzips a `.zip` file to a specified destination.

    Parameters
    ----------
    complete_file_path : str
        The full path to the `.zip` file.
    destination : str, optional
        The destination directory for the unzipped files. Default is the directory of the `.zip` file.

    Returns
    -------
    None
        This method does not return any values.

    Raises
    ------
    ValueError
        If the file path is invalid or does not end with `.zip`.
    """
    if complete_file_path is None or not complete_file_path.endswith('.zip'):
        raise ValueError('file_name ending with .zip needs to be provided...')

    if destination is None:
        destination = os.path.dirname(complete_file_path)

    print(f"Unzipping: {complete_file_path}")
    with zipfile.ZipFile(complete_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    print(f"Unzipped: {complete_file_path}")


def parameter_scheduler(strategy: str = 'half', parameter_list: list = None, lower_bound: int = 1):
    """
    Adjusts parameter values based on a specified strategy.

    Parameters
    ----------
    strategy : str, optional
        The scheduling strategy to apply. Supported: 'half'. Default is 'half'.
    parameter_list : list, optional
        A list of parameters to schedule. Default is None.
    lower_bound : int, optional
        The minimum value a parameter can take. Default is 1.

    Returns
    -------
    list
        The adjusted parameter values.

    Raises
    ------
    ValueError
        If `parameter_list` is not provided or the strategy is unsupported.
    """
    if parameter_list is None:
        raise ValueError("Parameter list must be provided.")
    if strategy == 'half':
        result = [max(int(parameter/2), lower_bound) if parameter is not None else None for parameter in parameter_list]
        return result
    else:
        raise ValueError("Parameter strategy not supported.")

