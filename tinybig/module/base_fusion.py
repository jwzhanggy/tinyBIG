# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

########################
# Fusion Function Base #
########################

from abc import abstractmethod

import torch
from torch.nn import Module

from tinybig.module.base_function import function
from tinybig.config import config


class fusion(Module, function):
    r"""
    A base class for fusion operations, extending the `Module` and `function` classes.

    This class provides mechanisms for preprocessing, postprocessing, and fusing input tensors.
    It allows the use of customizable functions for data transformations and facilitates
    the definition of fusion-specific parameters and methods.

    Notes
    ---------
    In the tinyBIG library, we introduce several advanced fusion strategies that can more effectively aggregate the outputs from the wide architectures.
    Formally, given the input matrices $\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k$, their fusion output can be represented as

    $$
        \begin{equation}
        \mathbf{A} = \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k).
        \end{equation}
    $$

    The dimensions of the input matrices $\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k$ may be identical or vary,
    depending on the specific definition of the fusion function.



    Parameters
    ----------
    dims : list[int] | tuple[int], optional
        A list or tuple of dimensions for the input tensors, by default None.
    name : str, optional
        The name of the fusion operation, by default 'base_fusion'.
    require_parameters : bool, optional
        Whether the fusion operation requires trainable parameters, by default False.
    preprocess_functions : list | tuple | callable, optional
        Functions to preprocess the input tensors, by default None.
    postprocess_functions : list | tuple | callable, optional
        Functions to postprocess the output tensors, by default None.
    preprocess_function_configs : dict, optional
        Configuration for instantiating the preprocess functions, by default None.
    postprocess_function_configs : dict, optional
        Configuration for instantiating the postprocess functions, by default None.
    device : str, optional
        The device for computations, by default 'cpu'.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.
    """
    def __init__(
        self,
        dims: list[int] | tuple[int] = None,
        name: str = 'base_fusion',
        require_parameters: bool = False,
        preprocess_functions=None,
        postprocess_functions=None,
        preprocess_function_configs=None,
        postprocess_function_configs=None,
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
        Initializes the fusion class with its parameters and preprocessing/postprocessing functions.

        Parameters
        ----------
        dims : list[int] | tuple[int], optional
            A list or tuple of dimensions for the input tensors, by default None.
        name : str, optional
            The name of the fusion operation, by default 'base_fusion'.
        require_parameters : bool, optional
            Whether the fusion operation requires trainable parameters, by default False.
        preprocess_functions : list | tuple | callable, optional
            Functions to preprocess the input tensors, by default None.
        postprocess_functions : list | tuple | callable, optional
            Functions to postprocess the output tensors, by default None.
        preprocess_function_configs : dict, optional
            Configuration for instantiating the preprocess functions, by default None.
        postprocess_function_configs : dict, optional
            Configuration for instantiating the postprocess functions, by default None.
        device : str, optional
            The device for computations, by default 'cpu'.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        Module.__init__(self)
        function.__init__(self, name=name, device=device)

        self.dims = dims
        self.require_parameters = require_parameters

        self.preprocess_functions = config.instantiation_functions(preprocess_functions, preprocess_function_configs, device=self.device)
        self.postprocess_functions = config.instantiation_functions(postprocess_functions, postprocess_function_configs, device=self.device)


    def get_dims(self):
        """
        Retrieves the dimensions of the input tensors.

        Returns
        -------
        list[int] | tuple[int] | None
            The dimensions of the input tensors, or None if not specified.
        """
        return self.dims

    def get_num(self):
        """
        Retrieves the number of dimensions.

        Returns
        -------
        int
            The number of dimensions, or 0 if `dims` is not specified.
        """
        if self.dims is not None:
            return len(self.dims)
        else:
            return 0

    def get_dim(self, index: int):
        """
        Retrieves the dimension at the specified index.

        Parameters
        ----------
        index : int
            The index of the dimension to retrieve.

        Returns
        -------
        int
            The dimension at the specified index.

        Raises
        ------
        ValueError
            If the index is out of bounds or `dims` is not specified.
        """
        if self.dims is not None:
            if index is not None and 0 <= index <= len(self.dims):
                return self.dims[index]
            else:
                raise ValueError(f'Index {index} is out of dim list bounds...')
        else:
            return None

    def pre_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        Applies preprocessing functions to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        device : str, optional
            The computational device, by default 'cpu'.

        Returns
        -------
        torch.Tensor
            The preprocessed tensor.
        """
        return function.func_x(x, self.preprocess_functions, device=device)

    def post_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        Applies postprocessing functions to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        device : str, optional
            The computational device, by default 'cpu'.

        Returns
        -------
        torch.Tensor
            The postprocessed tensor.
        """
        return function.func_x(x, self.postprocess_functions, device=device)

    def to_config(self):
        """
        Serializes the fusion instance into a configuration dictionary.

        Returns
        -------
        dict
            A dictionary containing the class name and parameters,
            along with serialized preprocessing and postprocessing function configurations.
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
    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        """
        Abstract method to calculate a value `n` based on dimensions or other parameters.

        Parameters
        ----------
        dims : list[int] | tuple[int], optional
            The input dimensions, by default None.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def calculate_l(self, *args, **kwargs):
        """
        Abstract method to calculate a value `l` based on specific parameters.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor], w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
        Abstract method to define the forward pass of the fusion operation.

        Parameters
        ----------
        x : list[torch.Tensor] | tuple[torch.Tensor]
            A list or tuple of input tensors.
        w : torch.nn.Parameter, optional
            Trainable parameters for the fusion operation, by default None.
        device : str, optional
            The computational device, by default 'cpu'.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses.
        """
        pass
