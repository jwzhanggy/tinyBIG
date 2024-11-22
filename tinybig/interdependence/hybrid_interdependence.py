# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Hybrid Interdependence #
##########################
"""
The hybrid interdependence functions

This module contains the hybrid interdependence function.
"""

from typing import Callable

import torch

from tinybig.interdependence import interdependence
from tinybig.config.base_config import config


class hybrid_interdependence(interdependence):
    r"""
        A hybrid interdependence class combining multiple interdependence functions.

        This class enables the combination of multiple interdependence functions and applies
        a fusion function to aggregate the results. It supports configurations for both
        interdependence functions and the fusion function.

        Notes
        ----------
        Formally, given the input data batch $\mathbf{X} \in R^{b \times m}$, we can define a set of data and
        structural interdependence functions $\xi_1, \xi_2, \cdots, \xi_k: R^{b \times m} \to R^{m \times m'}$
        to measure the interdependence relationships among the attributes. These functions can be effectively fused together as follows:

        $$
            \begin{equation}
            \begin{aligned}
            \xi(\mathbf{X}) &= \text{fusion} \left( \xi_1(\mathbf{X}), \xi_2(\mathbf{X}), \cdots, \xi_k(\mathbf{X}) \right)\\
            &= \text{fusion} \left( \mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k \right)\\
            &= \mathbf{A} \in R^{m \times m'},
            \end{aligned}
            \end{equation}
        $$

        where $\mathbf{A}_i = \xi_i(\mathbf{X})$ denotes the interdependence matrix obtained by function $\xi_i, \forall i \in \{1, 2, \cdots, k\}$.
        Different fusion strategies can be used to define the $\text{fusion}(\cdot)$ operator used above, which will be introduced in the following subsection specifically.

        Attributes
        ----------
        interdependence_functions : list
            List of interdependence function objects.
        fusion_function : Callable
            The fusion function used to combine the outputs of the interdependence functions.
        require_parameters : bool
            Indicates whether any interdependence or fusion function requires parameters.
        require_data : bool
            Indicates whether any interdependence or fusion function requires input data.

        Methods
        -------
        __init__(...)
            Initializes the hybrid interdependence function.
        get_function_parameter_numbers()
            Retrieves the number of parameters required by each interdependence function and the fusion function.
        calculate_l()
            Calculates the total number of parameters required for the hybrid interdependence.
        calculate_b_prime(b=None)
            Computes the number of rows in the output tensor after applying interdependence.
        calculate_m_prime(m=None)
            Computes the number of columns in the output tensor after applying interdependence.
        calculate_A(x=None, w=None, device='cpu', *args, **kwargs)
            Computes the hybrid interdependence matrix.
    """

    def __init__(
        self,
        b: int, m: int,
        interdependence_type: str = 'attribute',
        name: str = 'hybrid_interdependence',
        interdependence_functions: list = None,
        interdependence_function_configs: list = None,
        fusion_function: Callable = None,
        fusion_function_config: dict = None,
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
            Initializes the hybrid interdependence function.

            Parameters
            ----------
            b : int
                Number of rows in the input tensor.
            m : int
                Number of columns in the input tensor.
            interdependence_type : str, optional
                Type of interdependence ('attribute', 'instance', etc.). Defaults to 'attribute'.
            name : str, optional
                Name of the interdependence function. Defaults to 'hybrid_interdependence'.
            interdependence_functions : list, optional
                List of pre-initialized interdependence function objects. Defaults to None.
            interdependence_function_configs : list, optional
                List of configuration dictionaries for initializing interdependence functions. Defaults to None.
            fusion_function : Callable, optional
                Pre-initialized fusion function object. Defaults to None.
            fusion_function_config : dict, optional
                Configuration dictionary for initializing the fusion function. Defaults to None.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.

            Raises
            ------
            ValueError
                If neither `interdependence_functions` nor `interdependence_function_configs` are provided.
                If neither `fusion_function` nor `fusion_function_config` is provided.
        """

        if interdependence_functions is not None:
            self.interdependence_functions = interdependence_functions
            for func in self.interdependence_functions:
                func.b = b
                func.m = m
                func.interdependence_type = interdependence_type
        elif interdependence_function_configs is not None:
            self.interdependence_functions = []
            for function_config in interdependence_function_configs:
                assert 'function_class' in function_config
                function_class = function_config['function_class']
                function_parameters = function_config['function_parameters'] if 'function_parameters' in function_config else {}
                function_parameters['b'] = b
                function_parameters['m'] = m
                function_parameters['interdependence_type'] = interdependence_type
                self.interdependence_functions.append(config.get_obj_from_str(function_class)(**function_parameters))
        else:
            raise ValueError('No interdependence functions or configurations are specified...')

        require_parameters = any([func.require_parameters for func in self.interdependence_functions])
        require_data = any([func.require_data for func in self.interdependence_functions])
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_parameters=require_parameters, require_data=require_data, device=device, *args, **kwargs)

        if fusion_function is not None:
            self.fusion_function = fusion_function
        elif fusion_function_config is not None:
            assert 'function_class' in fusion_function_config
            function_class = fusion_function_config['function_class']
            function_parameters = fusion_function_config['function_parameters'] if 'function_parameters' in fusion_function_config else {}
            self.fusion_function = config.get_obj_from_str(function_class)(**function_parameters)
        else:
            raise ValueError('No fusion function or configurations are specified...')

    def get_function_parameter_numbers(self):
        """
            Retrieves the number of parameters required by each interdependence function and the fusion function.

            Returns
            -------
            list of int
                A list containing the number of parameters for each interdependence function and the fusion function.
        """
        l_list = []
        for func in self.interdependence_functions:
            if func.require_parameters:
                l_list.append(func.calculate_l())
            else:
                l_list.append(0)
        if self.fusion_function.require_parameters:
            l_list.append(self.fusion_function.calculate_l())
        else:
            l_list.append(0)
        return l_list

    def calculate_l(self):
        """
            Calculates the total number of parameters required for the hybrid interdependence.

            Returns
            -------
            int
                Total number of parameters required.
        """
        l_list = self.get_function_parameter_numbers()
        return sum(l_list)

    def calculate_b_prime(self, b: int = None):
        """
            Computes the number of rows in the output tensor after applying interdependence.

            Parameters
            ----------
            b : int, optional
                Number of rows in the input tensor. Defaults to `self.b`.

            Returns
            -------
            int
                The number of rows in the output tensor.
        """
        b_prime_list = [func.calculate_b_prime(b=b) for func in self.interdependence_functions]
        return self.fusion_function.calculate_n(dims=b_prime_list)

    def calculate_m_prime(self, m: int = None):
        """
            Computes the number of columns in the output tensor after applying interdependence.

            Parameters
            ----------
            m : int, optional
                Number of columns in the input tensor. Defaults to `self.m`.

            Returns
            -------
            int
                The number of columns in the output tensor.
        """
        m_prime_list = [func.calculate_b_prime(m=m) for func in self.interdependence_functions]
        return self.fusion_function.calculate_n(dims=m_prime_list)

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        r"""
            Computes the hybrid interdependence matrix.

            Formally, given the input data batch $\mathbf{X} \in R^{b \times m}$, we can define a set of data and
            structural interdependence functions $\xi_1, \xi_2, \cdots, \xi_k: R^{b \times m} \to R^{m \times m'}$
            to measure the interdependence relationships among the attributes. These functions can be effectively fused together as follows:

            $$
                \begin{equation}
                \begin{aligned}
                \xi(\mathbf{X}) &= \text{fusion} \left( \xi_1(\mathbf{X}), \xi_2(\mathbf{X}), \cdots, \xi_k(\mathbf{X}) \right)\\
                &= \text{fusion} \left( \mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k \right)\\
                &= \mathbf{A} \in R^{m \times m'},
                \end{aligned}
                \end{equation}
            $$

            where $\mathbf{A}_i = \xi_i(\mathbf{X})$ denotes the interdependence matrix obtained by the function $\xi_i, \forall i \in \{1, 2, \cdots, k\}$.
            Different fusion strategies can be used to define the $\text{fusion}(\cdot)$ operator used above, which will be introduced in the following subsection specifically.


            Parameters
            ----------
            x : torch.Tensor, optional
                Input tensor of shape `(batch_size, num_features)`. Defaults to None.
            w : torch.nn.Parameter, optional
                Parameter tensor. Required if interdependence or fusion functions need parameters. Defaults to None.
            device : str, optional
                Device for computation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments for interdependence functions.
            **kwargs : dict
                Additional keyword arguments for interdependence functions.

            Returns
            -------
            torch.Tensor
                The computed hybrid interdependence matrix.

            Raises
            ------
            AssertionError
                If the parameter tensor `w` does not match the required size.
            ValueError
                If neither input data nor parameters are provided when required.
        """
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            parameter_numbers = self.get_function_parameter_numbers()
            assert w.numel() == sum(parameter_numbers)

            x = self.pre_process(x=x, device=device)

            A_list = []
            sparse_tag = False
            if w is not None:
                w_segments = torch.split(w, parameter_numbers, dim=-1)
            else:
                w_segments = [None] * len(parameter_numbers)

            for func, w_segment in zip(self.interdependence_functions, w_segments):
                A = func.calculate_A(x=x, w=w_segment, device=device, *args, **kwargs)
                if A.is_sparse:
                    A = A.to_dense()
                    sparse_tag = True
                A_list.append(A)

            if self.fusion_function.require_parameters:
                A = self.fusion_function(x=A_list, w=w_segments[-1], device=device, *args, **kwargs)
            else:
                A = self.fusion_function(x=A_list, device=device, *args, **kwargs)

            A = self.post_process(x=A, device=device)

            if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
                assert A.shape == (self.m, self.calculate_m_prime())
            elif self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
                assert A.shape == (self.b, self.calculate_b_prime())
            if sparse_tag:
                A = A.to_sparse_coo()

            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A

            return A


