# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#########################
#  Transformation Base  #
#########################

"""
Base transformation function for data.

This module contains the base data transformation function class definition.
The data expansion and compression functions are all defined based on this transformation class.
"""

from abc import abstractmethod
import torch

from torch.nn import Module

from tinybig.module.base_function import function
from tinybig.config import config


class transformation(Module, function):
    r"""
    The base class of the data transformation function in the tinyBIG toolkit.

    It will be used as the base class template for defining the data expansion and compression functions.

    ...

    Notes
    ----------
    Formally, given the underlying data distribution mapping $f: {R}^m \to {R}^n$ to be learned,
    the data expansion function $\kappa$ projects input data into a new space shown as follows:

    $$ \kappa: {R}^m \to {R}^{D}, $$

    where the target dimension vector space dimension $D$ is determined when defining $\kappa$.

    In practice, the function $\kappa$ can either expand or compress the input to a higher- or lower-dimensional space.
    The corresponding function, $\kappa$, can also be referred to as the data expansion function (if $D > m$)
    and data compression function (if $D < m$), respectively. Collectively, these can be unified under the term
    "data transformation functions".

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
        Device of the data transformation function of the data transformation.

    Methods
    ----------
    __init__
        It initializes the data transformation function.

    pre_process
        It performs the pre-processing of the input data before transformation.

    post_process
        It performs the post-processing of the input data after transformation.

    calculate_D
        It calculate the expansion space dimension based on the input dimension parameter.

    forward
        The forward method to perform data transformation.

    __call__
        The built-in callable method of the data transformation function.
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
        The initialization method of the base data transformation function.

        It initializes a base data transformation function object.

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
            The base data transformation function object.
        """
        Module.__init__(self)
        function.__init__(self, name=name, device=device)

        self.preprocess_functions = config.instantiation_functions(preprocess_functions, preprocess_function_configs, device=self.device)
        self.postprocess_functions = config.instantiation_functions(postprocess_functions, postprocess_function_configs, device=self.device)
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
        return function.func_x(x, self.preprocess_functions, device=device)

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
        return function.func_x(x, self.postprocess_functions, device=device)

    def to_config(self):
        """
        Converts the current instance of the `interdependence` class into a configuration dictionary.

        This method generates a configuration dictionary containing the class name and the current
        attributes of the instance. The preprocessing and postprocessing functions are excluded
        from the attributes and replaced with their respective configuration details if available.

        Returns
        -------
        dict
            A dictionary representing the configuration of the `interdependence` instance with the following structure:
            {
                "function_class": str,
                    The fully qualified class name, including the module and class name.
                "function_parameters": dict,
                    The instance attributes as key-value pairs, excluding `preprocess_functions`
                    and `postprocess_functions`. If `preprocess_functions` or `postprocess_functions`
                    are defined, their configurations are included under `preprocess_function_configs`
                    and `postprocess_function_configs`, respectively.
            }

        See Also
        --------
        function.functions_to_configs : Converts functions or a list of functions into configuration dictionaries.
        """

        class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        attributes = {attr: getattr(self, attr) for attr in self.__dict__}
        attributes.pop('preprocess_functions')
        attributes.pop('postprocess_functions')

        if self.preprocess_functions is not None:
            attributes['preprocess_function_configs'] = function.functions_to_configs(self.preprocess_functions)
        if self.postprocess_functions is not None:
            attributes['postprocess_function_configs'] = function.functions_to_configs(self.postprocess_functions)

        return {
            "function_class": class_name,
            "function_parameters": attributes
        }

    @abstractmethod
    def calculate_D(self, m: int):
        """
        The transformation dimension calculation method.

        It calculates the intermediate transformation space dimension based on the input dimension parameter m.
        The method is declared as an abstractmethod and needs to be implemented in the inherited classes.

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

    def __call__(self, *args, **kwargs):
        """
        The re-implementation of the callable method.

        It applies the data expansion operation to the input data and returns the
        expansion result by calling the "forward" method.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        The forward method of the data transformation function.


        It applies the data expansion operation to the input data and returns the expansion result.
        The method is declared as a abstractmethod and needs to be implemented in the inherited classes.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device to perform the data transformation.

        Returns
        ----------
        torch.Tensor
            The expanded data vector of the input.
        """
        pass