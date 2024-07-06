# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
The RPN head with multi-channels.

This module contains the implementation of RPN head with multiple channels.
The RPN head will be used to compose the RPN layer module for building deep RPN models.
"""

import math
import torch
import torch.nn.functional as F

from tinybig.util import get_obj_from_str, process_function_list, func_x, register_function_parameters


class rpn_head(torch.nn.Module):
    r"""
    The RPN head class for implementing the multi-channel module.

    It will be used to compose the RPN layer module for building deep RPN models.

    ...

    Notes
    ----------
    Similar to convolutional neural networks (CNNs) employing multiple filters, RPN allows each head to have multiple
    channels of parameters applied to the same data expansion.
    RPN defines its multi-channel parameters as $\mathbf{w}^{0}, \mathbf{w}^{1}, \cdots, \mathbf{w}^{C-1}$,
    where $C$ denotes the number of channels.
    Based on the data expansion, parameter reconciliation and remainder functions, the RPN head will calculate its
    output with such multi-channel parameters as follows:
    $$
        \begin{equation}
            g(\mathbf{x} | \mathbf{w}, C) = \sum_{c=0}^{C-1} \left\langle \kappa(\mathbf{x}), \psi(\mathbf{w}^{c}) \right\rangle + \pi(\mathbf{x}),
        \end{equation}
    $$
    where these multi-channel parameters are fabricated from length $l$ to shape $(n, D)$ using the identical
    parameter reconciliation function.

    Attributes
    ----------
    m: int
        The input dimension of the head.
    n: int
        The output dimension of the head.
    l: int, default = None
        The number of parameter for each channel in the head.
    channel_num: int, default = 1
        The number of channels in the head.
    data_transformation: object, default = None
        The data transformation function of the head. The data transformation can be initialized directly
        with this parameter or with the data_transformation_config parameter.
    parameter_fabrication: object, default = None
        The parameter fabrication function of the head. The parameter fabrication can be initialized directly
        with this parameter or with the parameter_fabrication_config parameter.
    remainder: object, default = None
        The remainder function the head. The remainder can be initialized directly
        with this parameter or with the remainder_config parameter.
    w: torch.nn.Parameter, default = None
        The parameters used for the parameter reconciliation function of length $l$ for each channel,
        which will be fabricated into a parameter matrix of shape $(n, D)$.
    b: torch.nn.Parameter, default = None
        The (optional) bias parameters used for the parameter reconciliation function.
    w_prime: torch.nn.Parameter, default = None
        The (optional) parameters used for the remainder function.
    b_prime: torch.nn.Parameter, default = None
        The (optional) bias parameters used for the remainder function.
    device: str, default = 'cpu'
        The device for hosting the head.

    Methods
    ----------
    __init__
        The initialization method of the RPN-head with multiple channels.

    initialize_parameters
        The parameter initialization method.

    initialize_parameters_kaiming
        The kaiming parameter initialization method.

    initialize_parameters_xavier
        The xavier initialization method.

    output_process
        The output processing method of the head.

    forward
        The forward method of the RPN head module.

    __call__
        The re-implementation of the builtin callable method based on the forward method.
    """
    def __init__(
            self,
            m: int,
            n: int,
            l: int = None,
            channel_num: int = 1,
            data_transformation=None,
            parameter_fabrication=None,
            remainder=None,
            output_process_functions=None,
            data_transformation_configs=None,
            parameter_fabrication_configs=None,
            remainder_configs=None,
            output_process_function_configs=None,
            device='cpu',
            *args, **kwargs
    ):
        r"""
        The initialization method of the RPN-head with multiple channels.

        It initializes the RPN head module with multi-channel.
        Specifically, this method initializes the dimension configurations of the head,
        the component functions used in the head, and defines the device to host the head.

        Parameters
        ----------
        m: int
            The input dimension of the head.
        n: int
            The output dimension of the head.
        l: int, default = None
            The number of parameter for each channel in the head.
        channel_num: int, default = 1
            The number of channels in the head.
        data_transformation: object, default = None
            The data transformation function of the head. The data transformation can be initialized directly
            with this parameter or with the data_transformation_config parameter.
        parameter_fabrication: object, default = None
            The parameter fabrication function of the head. The parameter fabrication can be initialized directly
            with this parameter or with the parameter_fabrication_config parameter.
        remainder: object, default = None
            The remainder function the head. The remainder can be initialized directly
            with this parameter or with the remainder_config parameter.
        output_process_functions: object, default = None
            The output processing functions. The output processing function can be initialized directly
            with this parameter or with the output_process_function_configs parameter.
        data_transformation_configs: dict, default = None
            The data transformation function configuration.
        parameter_fabrication_configs: dict, default = None
            The parameter fabrication function configuration.
        remainder_configs: dict, default = None
            The remainder function configuration.
        output_process_function_configs: dict, default = None
            The output processing function configuration.
        device: str, default = 'cpu'
            The device for hosting the head.

        Returns
        ----------
        object
            This method will return the initialized RPN-head object.
        """
        super().__init__()
        assert channel_num >= 1 and m is not None and m >= 1 and n is not None and n >= 1

        # initialize the basic attributes
        self.m = m
        self.n = n
        self.channel_num = channel_num
        # l can be calculated with parameter_fabrication function, it can be none
        self.l = l
        self.device = device

        # initialize data_transformation, parameter_fabrication and remainder from either input objects or input configs
        if data_transformation is not None:
            self.data_transformation = data_transformation
        elif data_transformation_configs is not None:
            data_transformation_class = data_transformation_configs['data_transformation_class']
            parameters = data_transformation_configs['data_transformation_parameters'] if 'data_transformation_parameters' in data_transformation_configs else {}
            parameters['device'] = device
            self.data_transformation = get_obj_from_str(data_transformation_class)(**parameters)
        else:
            self.data_transformation = None

        if parameter_fabrication is not None:
            self.parameter_fabrication = parameter_fabrication
        elif parameter_fabrication_configs is not None:
            parameter_fabrication_class = parameter_fabrication_configs['parameter_fabrication_class']
            parameters = parameter_fabrication_configs['parameter_fabrication_parameters'] if 'parameter_fabrication_parameters' in parameter_fabrication_configs else {}
            parameters['device'] = device
            self.parameter_fabrication = get_obj_from_str(parameter_fabrication_class)(**parameters)
        else:
            self.parameter_fabrication = None

        if remainder is not None:
            self.remainder = remainder
        elif remainder_configs is not None:
            remainder_class = remainder_configs['remainder_class']
            parameters = remainder_configs['remainder_parameters'] if 'remainder_parameters' in remainder_configs else {}
            parameters['device'] = device
            self.remainder = get_obj_from_str(remainder_class)(**parameters)
        else:
            self.remainder = None

        # initialize the output processing functions from either function list or configs
        self.output_process_functions = process_function_list(output_process_functions, output_process_function_configs, device=device)
        #register_function_parameters(self, self.output_process_functions)

        # initialize parameters for parameter_fabrication function
        self.w = None
        self.b = None
        if self.parameter_fabrication is not None and self.parameter_fabrication.require_parameters:
            if self.l is None:
                self.l = self.parameter_fabrication.calculate_l(n=self.n, D=self.data_transformation.calculate_D(m=self.m))
            # channel_num*l, where l = nxD
            self.w = torch.nn.Parameter(torch.rand(1, self.l*self.channel_num))
            if self.parameter_fabrication.enable_bias:
                # [n]
                self.b = torch.nn.Parameter(torch.rand(self.n))

        # initialize the optional parameters for the remainder function
        if self.m != self.n and not self.remainder.require_parameters and self.remainder.get_name() not in ['zero_remainder', 'constant_remainder']:
            raise ValueError('The input and output dimensions {}, {} are different, parameters will be needed '
                             'by the {} to adjust the input dimensions.'.format(self.m, self.n, self.remainder.get_name()))

        self.w_prime = None
        self.b_prime = None
        if self.remainder is not None and self.remainder.require_parameters:
            # [n, m], w_prime.T will be multipled with x,
            self.w_prime = torch.nn.Parameter(torch.rand(self.n, self.m))
            if self.remainder.enable_bias:
                # [n]
                self.b_prime = torch.nn.Parameter(torch.rand(self.n))

    def initialize_parameters(self, init_type='xavier_uniform', init_bias=True, *args, **kwargs):
        """
        The parameter initialization method.

        It initializes the multi-channel parameters in the head with different initialization approaches,
        e.g., xavier_uniform or kaiming_uniform.
        Depending on the "init_type" parameter, this method will call the corresponding initiation methods.

        Parameters
        ----------
        init_type: str, default = 'xavier_uniform'
            The parameter initialization approach.
        init_bias: bool, default = True
            The boolean tag of bias initialization.

        Returns
        -------
        None
            This initialization method doesn't have any return values.
        """
        if init_type == 'kaiming_uniform':
            self.initialize_parameters_kaiming(init_bias=init_bias, *args, **kwargs)
        elif init_type == 'xavier_uniform':
            self.initialize_parameters_xavier(init_bias=init_bias, *args, **kwargs)

    def initialize_parameters_kaiming(self, init_bias=True, *args, **kwargs):
        """
        The kaiming parameter initialization method.

        It initializes the multi-channel parameters in the head with kaiming_uniform_ method from pytorch.

        Parameters
        ----------
        init_bias: bool, default = True
            The boolean tag of bias initialization.

        Returns
        -------
        None
            This initialization method doesn't have any return values.
        """
        if self.w is not None:
            torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.w_prime is not None:
            torch.nn.init.kaiming_uniform_(self.w_prime, a=math.sqrt(5))
        if init_bias:
            if self.b is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.b, -bound, bound)
            if self.b_prime is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w_prime)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.b_prime, -bound, bound)

    def initialize_parameters_xavier(self, init_bias=True, *args, **kwargs):
        """
        The xavier initialization method.

        It initializes the multi-channel parameters in the head with xavier_uniform_ method from pytorch.

        Parameters
        ----------
        init_bias: bool, default = True
            The boolean tag of bias initialization.

        Returns
        -------
        None
            This initialization method doesn't have any return values.
        """
        if self.w is not None:
            torch.nn.init.xavier_uniform_(self.w)
        if self.w_prime is not None:
            torch.nn.init.xavier_uniform_(self.w_prime)
        if init_bias:
            if self.b is not None:
                torch.nn.init.xavier_uniform_(self.b.view(1, -1))
            if self.b_prime is not None:
                torch.nn.init.xavier_uniform_(self.b_prime.view(1, -1))

    def output_process(self, x: torch.Tensor, *args, **kwargs):
        """
        The output processing method of the head.

        It processes the calculation output of the head with the output processing functions.
        The output processing functions are optional, which can be either an individual function or a list of functions.

        Parameters
        ----------
        x: torch.Tensor
            The calculation output of the head.

        Returns
        -------
        torch.Tensor
            The output processed with the output processing function.
        """
        return func_x(x, self.output_process_functions, device=self.device)

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the RPN head module.

        Based on the data expansion, parameter reconciliation and remainder functions, the RPN head will calculate its
        output with multi-channel parameters as follows:
        $$
            \begin{equation}
                g(\mathbf{x} | \mathbf{w}, C) = \sum_{c=0}^{C-1} \left\langle \kappa(\mathbf{x}), \psi(\mathbf{w}^{c}) \right\rangle + \pi(\mathbf{x}),
            \end{equation}
        $$
        where these multi-channel parameters $\mathbf{w}^{0}, \mathbf{w}^{1}, \cdots, \mathbf{w}^{C-1}$ are fabricated
        from length $l$ to shape $(n, D)$ using the identical parameter reconciliation function.

        Parameters
        ----------
        x: torch.Tensor
            The input data vector.
        device: str, default = 'cpu'
            The device for hosting the head.

        Returns
        -------
        torch.Tensor
            The processed output of the head.
        """
        # [batch, m] -> [batch, D]
        self.data_transformation.to(device)
        self.parameter_fabrication.to(device)
        self.remainder.to(device)

        kappa_x = self.data_transformation(x, device=device)

        assert kappa_x.shape[1] == self.data_transformation.calculate_D(m=x.shape[1])
        D = kappa_x.shape[1]
        # [batch, D]
        result = kappa_x
        if self.parameter_fabrication is not None:
            # [l] -> [n, D]
            w_chunks = [None]*self.channel_num
            if self.w is not None:
                w_chunks = self.w.chunk(self.channel_num, dim=1)
            phi_w = []
            for channel_index in range(self.channel_num):
                phi_w.append(self.parameter_fabrication(w=w_chunks[channel_index], n=self.n, D=D, device=device))
            phi_w = torch.stack(phi_w, dim=0)
            # [channel_num, n, D] -> [n, D]
            phi_w = torch.mean(phi_w, dim=0)

            # [batch, D] x [n, D]^T + [n] -> [batch, n]
            result = F.linear(result, phi_w, bias=self.b)

        if self.remainder is not None:
            # [batch, m] x [n, m]^T + [n] -> [batch, n]
            pi_x = self.remainder(x=x, w=self.w_prime, b=self.b_prime, m=self.m, n=self.n, device=device).to(result.device)
            # [batch, n]
            result += pi_x

        # optional: [batch, n] -> [batch, n]
        result = self.output_process(x=result)

        # [batch, n]
        return result

    def __call__(self, *args, **kwargs):
        """
        The re-implementation of the builtin callable method based on the forward method.

        It re-implements the callable method of the head, which will call the "forward" method to calculate the output
        with the multi-channel RPN head module.

        Returns
        -------
        torch.Tensor
            The processed output of the head.
        """
        return self.forward(*args, **kwargs)
