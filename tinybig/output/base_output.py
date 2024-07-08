# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliatoutputn: IFM Lab, UC Davis
"""
The base output class.

This class implements the base class for RPN model output saving and loading.
It will also be used as the template for other future different task output processing as well in the tinyBIG toolkit.
"""

import pickle
from tinybig.util.util import create_directory_if_not_exists


class output:
    """
    The base output class.

    It implements the base output processing class,
    which will handle the model output saving and loading.

    Attributes
    ----------
    name: str, default = 'base_output'
        Name of the output class object.

    Methods
    ----------
    __init__
        The initializatoutputn method of the base output class.

    save
        The output saving method of the output class.
        This method is declared to be a static method.

    load
        The output laoding method of the output class.
        This method is declared to be a static method.
    """
    def __init__(self, name='base_output', *args, **kwargs):
        """
        The base output class initializatoutputn method.

        It initializes the base output class object for handling the output saving and loading.

        Parameters
        ----------
        name: str, default = 'base_output'
            Name of the base output class object.
        """
        self.name = name

    @staticmethod
    def save(result, cache_dir='./result', output_file='output', *args, **kwargs):
        """
        The output saving method.

        It saves the provided outputs to a binary file via pick.dump.

        Parameters
        ----------
        result
            The output results to be saved.
        cache_dir: str, default = './result'
            Directory to save the output result.
        output_file: str, default = 'output'
            Name of file to save the output result.

        Returns
        -------
        None
            This output result saving method doesn't have return values.
        """
        create_directory_if_not_exists(f"{cache_dir}/{output_file}")
        with open(f"{cache_dir}/{output_file}", 'wb') as f:
            pickle.dump(result, f)

    @staticmethod
    def load(cache_dir='./result', output_file='output', *args, **kwargs):
        """
        The output result loading method.

        It loads the output results from a binary file via the pick.load.

        Parameters
        ----------
        cache_dir: str, default = './result'
            Directory of the output result file.
        output_file: str, default = 'output'
            Name of file of the output result.

        Returns
        -------
        None
            The loaded output result.
        """
        with open("{}/{}".format(cache_dir, output_file), 'rb') as f:
            result = pickle.load(f)
        return result
