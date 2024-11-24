# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################################
# Interdependence Function Base #
#################################

import warnings
from abc import abstractmethod

import torch
from torch.nn import Module

from tinybig.module.base_function import function
from tinybig.config import config


class interdependence(Module, function):
    r"""
    A base class for interdependence computations, extending `Module` and `function`.

    This class defines the structure for interdependence calculations, including attribute
    and instance interdependencies. It supports preprocessing and postprocessing of data,
    as well as customizable interdependence configurations.

    Notes
    ---------
    Formally, given an input data batch $\mathbf{X} \in {R}^{b \times m}$ (with $b$ instances and each instance with $m$ attributes),
    the attribute and instance data interdependence functions are defined as:

    $$
        \begin{equation}
        \xi_a: {R}^{b \times m} \to {R}^{m \times m'} \text{, and }
        \xi_i: {R}^{b \times m} \to {R}^{b \times b'},
        \end{equation}
    $$

    where $m'$ and $b'$ denote the output dimensions of their respective interdependence functions, respectively.

    Parameters
    ----------
    b : int
        The number of rows (e.g., instances) in the data.
    m : int
        The number of columns (e.g., attributes) in the data.
    name : str, optional
        The name of the interdependence operation, by default 'base_interdependency'.
    interdependence_type : str, optional
        The type of interdependence, e.g., 'attribute' or 'instance', by default 'attribute'.
    require_data : bool, optional
        Whether the operation requires input data, by default True.
    require_parameters : bool, optional
        Whether the operation requires trainable parameters, by default False.
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
        b: int, m: int,
        name: str = 'base_interdependency',
        interdependence_type: str = 'attribute',
        require_data: bool = True,
        require_parameters: bool = False,
        preprocess_functions=None,
        postprocess_functions=None,
        preprocess_function_configs=None,
        postprocess_function_configs=None,
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
        Initializes an instance of the `interdependence` class.

        The `interdependence` class is designed to model and process relationships between data features
        or instances (rows or columns) using a specific type of interdependence. The class supports
        preprocessing and postprocessing functions, and allows for customizable configurations.

        Parameters
        ----------
        b : int
            The number of rows (instances) in the input data.
        m : int
            The number of columns (attributes) in the input data.
        name : str, optional
            A name for the interdependence instance, by default 'base_interdependency'.
        interdependence_type : str, optional
            The type of interdependence. Valid values include:
            - 'row', 'left', 'instance', 'instance_interdependence' for instance-based operations.
            - 'column', 'right', 'attribute', 'attribute_interdependence' for attribute-based operations.
            By default, 'attribute'.
        require_data : bool, optional
            Specifies whether the `forward` method requires data input (`x`), by default True.
        require_parameters : bool, optional
            Specifies whether the `forward` method requires parameter input (`w`), by default False.
        preprocess_functions : list[Callable] | tuple[Callable], optional
            A list or tuple of preprocessing functions to apply before the primary operation, by default None.
        postprocess_functions : list[Callable] | tuple[Callable], optional
            A list or tuple of postprocessing functions to apply after the primary operation, by default None.
        preprocess_function_configs : dict, optional
            Configuration dictionary for instantiating preprocessing functions, by default None.
        postprocess_function_configs : dict, optional
            Configuration dictionary for instantiating postprocessing functions, by default None.
        device : str, optional
            The device to use for computations (e.g., 'cpu', 'cuda'), by default 'cpu'.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Attributes
        ----------
        name : str
            The name of the interdependence instance.
        interdependence_type : str
            The specified type of interdependence.
        b : int
            The number of rows (instances) in the input data.
        m : int
            The number of columns (attributes) in the input data.
        require_data : bool
            Whether the `forward` method requires data input.
        require_parameters : bool
            Whether the `forward` method requires parameter input.
        preprocess_functions : list[Callable]
            Instantiated preprocessing functions.
        postprocess_functions : list[Callable]
            Instantiated postprocessing functions.
        A : torch.Tensor or None
            The interdependence matrix `A`, initialized to None.
        device : str
            The computation device.

        Raises
        ------
        ValueError
            If the specified `interdependence_type` is invalid.
        """
        Module.__init__(self)
        function.__init__(self, name=name, device=device)

        self.interdependence_type = interdependence_type

        self.b = b
        self.m = m

        self.require_data = require_data
        self.require_parameters = require_parameters

        self.preprocess_functions = config.instantiation_functions(preprocess_functions, preprocess_function_configs, device=self.device)
        self.postprocess_functions = config.instantiation_functions(postprocess_functions, postprocess_function_configs, device=self.device)

        self.A = None

    @property
    def interdependence_type(self):
        """
        Retrieves the current interdependence type.

        Returns
        -------
        str
            The interdependence type.
        """
        return self._interdependence_type

    @interdependence_type.setter
    def interdependence_type(self, value):
        """
        Sets the type of interdependence and validates its value.

        The interdependence type determines whether the operation applies to rows
        (instances) or columns (attributes). Acceptable values are:
        - 'row', 'left', 'instance', 'instance_interdependence' for row-based operations
        - 'column', 'right', 'attribute', 'attribute_interdependence' for column-based operations.

        Parameters
        ----------
        value : str
            The type of interdependence to set.

        Raises
        ------
        ValueError
            If the provided value is not one of the allowed types.
        """
        allowed_values = ['instance_interdependence', 'instance', 'left', 'attribute_interdependence', 'attribute', 'right']
        if value not in allowed_values:
            raise ValueError(f"Invalid value for my_string. Allowed values are: {allowed_values}")
        self._interdependence_type = value

    def check_A_shape_validity(self, A: torch.Tensor):
        """
        Checks whether the shape of the interdependence matrix `A` is valid.

        The validity of the shape is determined by the current `interdependence_type`:
        - For row-based types, `A` should have shape `(b, b')`.
        - For column-based types, `A` should have shape `(m, m')`.

        Parameters
        ----------
        A : torch.Tensor
            The interdependence matrix to validate.

        Raises
        ------
        ValueError
            If `A` is None or its shape does not match the expected dimensions
            based on `b` or `m` and the `interdependence_type`.
        AssertionError
            If the provided matrix `A` does not meet shape requirements.
        """
        if A is None:
            raise ValueError("A must be provided")

        assert self.interdependence_type is not None and isinstance(self.interdependence_type, str)

        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert self.b is not None
            assert A.shape == (self.b, self.calculate_b_prime(b=self.b))
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert self.m is not None
            assert A.shape == (self.m, self.calculate_m_prime(m=self.m))
        else:
            raise ValueError("The interdependence type {self.interdependence_type} is not supported...}")

    def get_A(self):
        """
        Retrieves the current interdependence matrix `A`.

        Returns
        -------
        torch.Tensor or None
            The current interdependence matrix `A`, or None if `A` is not set.

        Warnings
        --------
        UserWarning
            If `A` is not set, a warning is issued.
        """
        if self.A is None:
            warnings.warn("The A matrix is None...")
            return None
        else:
            return self.A

    def get_b(self):
        """
        Retrieves the batch number (`b`).

        Returns
        -------
        int
            The batch size (i.e., the number of rows).
        """
        return self.b

    def get_m(self):
        """
        Retrieves the attribute number (`m`).

        Returns
        -------
        int
            The number of attributes (i.e., columns).
        """
        return self.m

    def pre_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        Applies the preprocessing functions to the input tensor.

        This method utilizes the functions specified in `preprocess_functions` to
        transform the input tensor before further processing.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to preprocess.
        device : str, optional
            The device to execute preprocessing on, by default 'cpu'.

        Returns
        -------
        torch.Tensor
            The preprocessed tensor.

        Notes
        -----
        - Preprocessing functions are instantiated during class initialization.
        - The exact transformations depend on the specified functions.
        """
        return function.func_x(x, self.preprocess_functions, device=device)

    def post_process(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        Applies the postprocessing functions to the output tensor.

        This method utilizes the functions specified in `postprocess_functions` to
        transform the output tensor after the primary computation.

        Parameters
        ----------
        x : torch.Tensor
            The output tensor to postprocess.
        device : str, optional
            The device to execute postprocessing on, by default 'cpu'.

        Returns
        -------
        torch.Tensor
            The postprocessed tensor.

        Notes
        -----
        - Postprocessing functions are instantiated during class initialization.
        - The exact transformations depend on the specified functions.
        """
        return function.func_x(x, self.postprocess_functions, device=device)

    def to_config(self):
        """
        Serializes the interdependence instance into a configuration dictionary.

        Returns
        -------
        dict
            A dictionary containing the class name, parameters, and configurations.
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

    def calculate_l(self):
        """
        Placeholder for calculating the learnable parameter number `l`.

        Returns
        -------
        int
            The learnable parameter number `l` (default is 0).
        """
        return 0

    def calculate_b_prime(self, b: int = None):
        """
        Calculates the transformed batch `b` dimension based on the interdependence type.

        Parameters
        ----------
        b : int, optional
            The original batch `b` dimension, by default None.

        Returns
        -------
        int
            The transformed batch `b` dimension.

        Warnings
        --------
        UserWarning
            If the interdependence type does not involve instances.
        """
        b = b if b is not None else self.b
        if self.interdependence_type not in ['row', 'left', 'instance', 'instance_interdependence']:
            warnings.warn("The interdependence_type is not about the instances, its b dimension will not be changed...")
        return b

    def calculate_m_prime(self, m: int = None):
        """
        Calculates the transformed attribute `m` dimension based on the interdependence type.

        Parameters
        ----------
        m : int, optional
            The original attribute `m` dimension, by default None.

        Returns
        -------
        int
            The transformed attribute `m` dimension.

        Warnings
        --------
        UserWarning
            If the interdependence type does not involve attributes.
        """
        m = m if m is not None else self.m
        if self.interdependence_type not in ['column', 'right', 'attribute', 'attribute_interdependence']:
            warnings.warn("The interdependence_type is not about the attributes, its m dimension will not be changed...")
        return m

    def forward(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, kappa_x: torch.Tensor = None, device: str = 'cpu', *args, **kwargs):
        """
        Executes the forward pass for the interdependence operation.

        Depending on the `interdependence_type`, the method calculates the transformation
        of the input tensor `x` or `kappa_x` using the interdependence matrix `A`.

        Parameters
        ----------
        x : torch.Tensor, optional
            The input data tensor, by default None.
        w : torch.nn.Parameter, optional
            Trainable parameters for the interdependence calculation, by default None.
        kappa_x : torch.Tensor, optional
            Alternative input tensor for processing, by default None.
        device : str, optional
            The device for computations, by default 'cpu'.

        Returns
        -------
        torch.Tensor
            The transformed data tensor.

        Raises
        ------
        AssertionError
            If `x` or `w` is required but not provided, or their dimensions are incorrect.
        ValueError
            If the `interdependence_type` is invalid or unsupported.

        Notes
        -----
        - For instance-based interdependence types, the operation transforms rows.
        - For attribute-based interdependence types, the operation transforms columns.
        """
        if self.require_data:
            assert x is not None and x.ndim == 2
        if self.require_parameters:
            assert w is not None and w.ndim == 2

        data_x = kappa_x if kappa_x is not None else x
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            # A shape: b * b'
            A = self.calculate_A(x.transpose(0, 1), w, device=device)
            assert A is not None and A.size(0) == data_x.size(0)
            if data_x.is_sparse or A.is_sparse:
                xi_x = torch.sparse.mm(A.t(), data_x)
            else:
                xi_x = torch.matmul(A.t(), data_x)
            return xi_x
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            # A shape: m * m'
            A = self.calculate_A(x, w, device)
            assert A is not None and A.size(0) == data_x.size(1)
            if data_x.is_sparse or A.is_sparse:
                xi_x = torch.sparse.mm(data_x, A)
            else:
                xi_x = torch.matmul(data_x, A)
            return xi_x
        else:
            raise ValueError(f"Invalid interdependence type: {self.interdependence_type}")


    @abstractmethod
    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
        Abstract method to calculate the interdependence matrix `A`.

        Parameters
        ----------
        x : torch.Tensor, optional
            The input data tensor, by default None.
        w : torch.nn.Parameter, optional
            The trainable parameter matrix, by default None.
        device : str, optional
            The computational device, by default 'cpu'.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses.
        """
        pass

