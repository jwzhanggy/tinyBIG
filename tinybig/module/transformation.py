# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Base transformation function.

This module contains the base transformation function class definition.
The data expansion functions
"""

from abc import abstractmethod
import torch

from tinybig.util import func_x, process_function_list, register_function_parameters

#########################
#  Transformation Base  #
#########################


class transformation(torch.nn.Module):
    """
    class transformation

    The base class of the data transformation function in the tinyBIG toolkit.

    ...

    Attributes
    ----------
    name: str, default = 'base_transformation'
        Name of the data transformation function.
    preprocess_functions: function | list, default = None
        Preprocessing function or function list.
    postprocess_functions: function | list, default = None
        Postprocessing function or function list.
    preprocess_function_configs: dict, default = None
        Configs of preprocessing function or function list.
    postprocess_function_configs: dict, default = None
        Configs of postprocessing function or function list.
    device: str, default = 'cpu'
        The device of the transformation function.

    Methods
    ----------
    __init__
        It performs the initialization of the data transformation function.

    get_name
        It gets the name of the data transformation function.

    pre_process
        It performs the pre-processing of the input data before transformation.

    post_process
        It performs the post-processing of the input data after transformation.

    calculate_D
        It calculate the expansion space dimension based on the input dimension parameter.

    __call__
        It reimplementation the build-in callable method.
    """

    def __init__(
            self,
            name='base_transformation',
            preprocess_functions=None,
            postprocess_functions=None,
            preprocess_function_configs=None,
            postprocess_function_configs=None,
            device='cpu',
            *args, **kwargs
    ):
        """
        The initialization method of the base transformation function.

        It initializes a base transformation function object based on the provided method parameters.

        Parameters
        ----------
        name: str, default = 'base_transformation'
            Name of the data transformation function.
        preprocess_functions: function | list, default = None
            Preprocessing function or function list.
        postprocess_functions: function | list, default = None
            Postprocessing function or function list.
        preprocess_function_configs: dict, default = None
            Configs of preprocessing function or function list.
        postprocess_function_configs: dict, default = None
            Configs of postprocessing function or function list.
        device: str, default = 'cpu'
            The device of the transformation function.

        Returns
        ----------
        object
            The base transformation function object.
        """
        super().__init__()
        self.name = name
        self.device = device
        self.preprocess_functions = process_function_list(preprocess_functions, preprocess_function_configs, device=self.device)
        self.postprocess_functions = process_function_list(postprocess_functions, postprocess_function_configs, device=self.device)
        # register_function_parameters(self, self.preprocess_functions)
        # register_function_parameters(self, self.postprocess_functions)

    def get_name(self):
        """
        The name retrieval method of data transformation function.

        It returns the name of the data transformation function.

        Returns
        -------
        str
            The name of the data transformation function.
        """
        return self.name

    def pre_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        The pre-processing method of data transformation function.

        It pre-process the input vector x with the (optional) pre-processing functions.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.
        args: list, default = ()
            The other parameters of the method.
        kwargs: dict, default = {}
            The other parameters of the method.

        Returns
        -------
        Tensor
            It returns the data vector after the pre-processing.
        """
        return func_x(x, self.preprocess_functions, device=device)

    def post_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        The post-processing method of data transformation function.

        It post-process the input vector x with the (optional) post-processing functions.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data expansion.
        args: list, default = ()
            The other parameters of the method.
        kwargs: dict, default = {}
            The other parameters of the method.

        Returns
        -------
        Tensor
            It returns the data vector after the post-processing.
        """
        return func_x(x, self.postprocess_functions, device=device)

    @abstractmethod
    def calculate_D(self, m: int):
        """
        The transformation dimension calculation method.

        It calculates the intermediate transformation space dimension based on the input dimension parameter m.
        The method is declared as a abstractmethod and needs to be implemented in the inherited classes.

        Parameters
        ----------
        m: int
            The dimension of the input space.

        Returns
        -------
        int
            The dimension of the transformation space.
        """
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        The callable method of the data transformation function.


        It executes the data expansion operation on the input data and returns the expansion result.
        The method is declared as a abstractmethod and needs to be implemented in the inherited classes.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data transformation.
        args: list, default = ()
            The other parameters.
        kwargs: dict, default = {}
            The other parameters.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        pass