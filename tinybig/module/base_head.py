# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

######################################
# RPN Multi-Channel Base Head Module #
######################################

"""
The RPN head with multi-channels.

This module contains the implementation of RPN head with multiple channels.
The RPN head will be used to compose the RPN layer module for building deep RPN models.
"""

import time
import math
import torch
import torch.nn.functional as F

from tinybig.config import config
from tinybig.module.base_functions import function
from tinybig.fusion.metric_fusion import mean_fusion
from tinybig.module import (
    transformation as transformation_class,
    fabrication as fabrication_class,
    remainder as remainder_class,
    interdependence as interdependence_class,
    fusion as fusion_class,
)


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
    w_remainder: torch.nn.Parameter, default = None
        The (optional) parameters used for the remainder function.
    b_remainder: torch.nn.Parameter, default = None
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

    output_processing
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
        batch_num: int = None,
        channel_num: int = 1,
        l: int = None,
        l_attribute_interdependence: int = None,
        l_instance_interdependence: int = None,
        l_channel_fusion: int = None,

        input_process_functions=None,
        data_transformation: transformation_class = None,
        attribute_interdependence: interdependence_class = None,
        instance_interdependence: interdependence_class = None,
        parameter_fabrication: fabrication_class = None,
        channel_fusion: fusion_class = None,
        remainder: remainder_class = None,
        output_process_functions=None,

        input_process_function_configs=None,
        data_transformation_configs=None,
        attribute_interdependence_configs=None,
        instance_interdependence_configs=None,
        parameter_fabrication_configs=None,
        channel_fusion_configs=None,
        remainder_configs=None,
        output_process_function_configs=None,

        create_parameters_at_init: bool = True,
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
            with this parameter or with the output_processing_function_configs parameter.
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
        assert (channel_num >= 1) and (m is not None and m >= 1) and (n is not None and n >= 1)

        # initialize the basic attributes
        self.m = m
        self.n = n
        self.batch_num = batch_num
        self.channel_num = channel_num
        self.l = l
        self.l_attribute_interdependence = l_attribute_interdependence
        self.l_instance_interdependence = l_instance_interdependence
        self.l_channel_fusion = l_channel_fusion
        self.device = device

        # initialize data_transformation, interdependence, interdependence_fusion, parameter_fabrication, channel_fusion and remainder functions from either input objects or input configs
        self.data_transformation = config.instantiation_functions(functions=data_transformation, function_configs=data_transformation_configs, device=device)
        self.parameter_fabrication = config.instantiation_functions(functions=parameter_fabrication, function_configs=parameter_fabrication_configs, device=device)
        self.remainder = config.instantiation_functions(functions=remainder, function_configs=remainder_configs, device=device)

        self.attribute_interdependence = config.instantiation_functions(functions=attribute_interdependence, function_configs=attribute_interdependence_configs, device=device)
        self.instance_interdependence = config.instantiation_functions(functions=instance_interdependence, function_configs=instance_interdependence_configs, device=device)

        self.input_process_functions = config.instantiation_functions(input_process_functions, input_process_function_configs, device=device)
        self.output_process_functions = config.instantiation_functions(output_process_functions, output_process_function_configs, device=device)

        self.channel_fusion = config.instantiation_functions(functions=channel_fusion, function_configs=channel_fusion_configs, device=device)
        if self.channel_num > 1 and self.channel_fusion is None:
            self.channel_fusion = mean_fusion(dims=[self.n] * self.channel_num)

        # create learnable parameters for parameter fabrication and remainder functions
        self.w = None
        self.b = None
        self.w_remainder = None
        self.b_remainder = None
        self.w_attribute_interdependence = None
        self.w_instance_interdependence = None
        self.w_channel_fusion = None

        if create_parameters_at_init:
            self.create_learnable_parameters()

    def get_m(self):
        return self.m

    def get_n(self):
        return self.n

    def get_channel_num(self):
        return self.channel_num

    def get_batch_num(self):
        return self.batch_num

    def create_learnable_parameters(
        self,
        initialize_parameter_at_creation: bool = False,
        init_type='xavier_uniform',
        init_bias=True,
        *args, **kwargs
    ):
        m_prime, b_prime = self.m, self.batch_num

        if self.attribute_interdependence is not None:
            if self.attribute_interdependence.require_parameters:
                if self.l_attribute_interdependence is None:
                    self.l_attribute_interdependence = self.attribute_interdependence.calculate_l()
                self.w_attribute_interdependence = torch.nn.Parameter(torch.rand(self.channel_num, self.l_attribute_interdependence, device=self.device))
            assert self.m is not None and self.m >= 1
            m_prime = self.attribute_interdependence.calculate_m_prime(m=self.m)

        if self.instance_interdependence is not None:
            if self.instance_interdependence.require_parameters:
                if self.l_instance_interdependence is None:
                    self.l_instance_interdependence = self.instance_interdependence.calculate_l()
                self.w_instance_interdependence = torch.nn.Parameter(torch.rand(self.channel_num, self.l_instance_interdependence, device=self.device))
            if self.batch_num is not None:
                assert self.batch_num is not None and self.batch_num >= 1
                b_prime = self.instance_interdependence.calculate_b_prime(b=self.batch_num)

        # create learnable parameters for parameter_fabrication function
        if self.parameter_fabrication is not None and self.parameter_fabrication.require_parameters:
            if self.l is None:
                self.l = self.parameter_fabrication.calculate_l(n=self.n, D=self.data_transformation.calculate_D(m=m_prime))
            self.w = torch.nn.Parameter(torch.rand(self.channel_num, self.l, device=self.device))
            if self.parameter_fabrication.enable_bias:
                self.b = torch.nn.Parameter(torch.rand(self.n, device=self.device))

        # create learnable parameters for remainder function
        if self.remainder is not None and self.remainder.require_parameters:
            self.w_remainder = torch.nn.Parameter(torch.rand(self.n, self.m, device=self.device))
            if self.remainder.enable_bias:
                self.b_remainder = torch.nn.Parameter(torch.rand(self.n, device=self.device))
        elif self.m != self.n and not self.remainder.require_parameters and self.remainder.get_name() not in ['zero_remainder', 'constant_remainder']:
            raise ValueError('The input and output dimensions {}, {} are different, parameters will be needed '
                             'by the {} to adjust the input dimensions.'.format(self.m, self.n, self.remainder.get_name()))

        # create learnable parameters for channel_fusion function
        if self.channel_fusion is not None and self.channel_fusion.require_parameters:
            if self.l_channel_fusion is None:
                self.l_channel_fusion = self.channel_fusion.calculate_l()
            self.w_channel_fusion = torch.nn.Parameter(torch.rand(1, self.l_channel_fusion, device=self.device))

        # initialize the parameter with certain methods...
        if initialize_parameter_at_creation:
            self.initialize_parameters(init_type=init_type, init_bias=init_bias)

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

        if self.w_attribute_interdependence is not None:
            torch.nn.init.kaiming_uniform_(self.w_attribute_interdependence, a=math.sqrt(5))

        if self.w_instance_interdependence is not None:
            torch.nn.init.kaiming_uniform_(self.w_instance_interdependence, a=math.sqrt(5))

        if self.w is not None:
            torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

        if self.w_remainder is not None:
            torch.nn.init.kaiming_uniform_(self.w_remainder, a=math.sqrt(5))

        if self.w_channel_fusion is not None:
            torch.nn.init.kaiming_uniform_(self.w_channel_fusion, a=math.sqrt(5))

        if init_bias:
            if self.b is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.b, -bound, bound)
            if self.b_remainder is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w_remainder)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.b_remainder, -bound, bound)

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
        if self.w_attribute_interdependence is not None:
            torch.nn.init.xavier_uniform_(self.w_attribute_interdependence)

        if self.w_instance_interdependence is not None:
            torch.nn.init.xavier_uniform_(self.w_instance_interdependence)

        if self.w is not None:
            torch.nn.init.xavier_uniform_(self.w)

        if self.w_remainder is not None:
            torch.nn.init.xavier_uniform_(self.w_remainder)

        if self.w_channel_fusion is not None:
            torch.nn.init.xavier_uniform_(self.w_channel_fusion)

        if init_bias:
            if self.b is not None:
                torch.nn.init.xavier_uniform_(self.b.view(1, -1))
            if self.b_remainder is not None:
                torch.nn.init.xavier_uniform_(self.b_remainder.view(1, -1))

    def to_config(self):
        head_class = f"{self.__class__.__module__}.{self.__class__.__name__}"
        head_parameters = {
            'name': self.name,
            'device': self.device,
            'm': self.m,
            'n': self.n,
            'l': self.l,
            'batch_num': self.batch_num,
            'channel_num': self.channel_num,
        }

        if self.data_transformation is not None:
            head_parameters['data_transformation_configs'] = self.data_transformation.to_config()
        if self.attribute_interdependence is not None:
            head_parameters['attribute_interdependence_configs'] = self.attribute_interdependence.to_config()
        if self.instance_interdependence is not None:
            head_parameters['instance_interdependence_configs'] = self.instance_interdependence.to_config()
        if self.parameter_fabrication is not None:
            head_parameters['parameter_fabrication_configs'] = self.parameter_fabrication.to_config()
        if self.channel_fusion is not None:
            head_parameters['channel_fusion_configs'] = self.channel_fusion.to_config()
        if self.remainder is not None:
            head_parameters['remainder_configs'] = self.remainder.to_config()
        if self.input_process_functions is not None:
            head_parameters['input_process_function_configs'] = function.functions_to_configs(self.input_process_functions)
        if self.output_process_functions is not None:
            head_parameters['output_process_function_configs'] = function.functions_to_configs(self.output_process_functions)

        return {
            "head_class": head_class,
            "head_parameters": head_parameters
        }

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

    def calculate_kappa_x(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        if self.data_transformation is not None:
            if self.data_transformation.device != device:
                self.data_transformation.to(device)

            kappa_x = self.data_transformation(x, device=device)
            return kappa_x
        else:
            return x

    def calculate_phi_w(self, D: int, channel_index: int = 0, device='cpu', *args, **kwargs):
        assert channel_index in range(self.channel_num)

        if self.parameter_fabrication is not None:
            if self.parameter_fabrication.device != device:
                self.parameter_fabrication.to(device)

            if self.w is not None and 0 <= channel_index < self.w.size(0):
                w_chunk = self.w[channel_index:channel_index+1, :]
            else:
                w_chunk = None
            phi_w = self.parameter_fabrication(w=w_chunk, n=self.n, D=D, device=device)
            return phi_w
        else:
            return None

    def calculate_pi_x(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        if self.remainder is not None:
            if self.remainder.device != device:
                self.remainder.to(device)
            pi_x = self.remainder(x=x, w=self.w_remainder, b=self.b_remainder, m=self.m, n=self.n, device=device)
            return pi_x
        else:
            return None

    def calculate_attribute_xi_x(self, x: torch.Tensor, channel_index: int = 0, kappa_x: torch.Tensor = None, device='cpu', *args, **kwargs):
        if self.attribute_interdependence is not None:
            if self.attribute_interdependence.device != device:
                self.attribute_interdependence.to(device)

            if self.w_attribute_interdependence is not None and 0 <= channel_index < self.w_attribute_interdependence.size(0):
                w_chunks = self.w_attribute_interdependence[channel_index:channel_index+1, :]
            else:
                w_chunks = None
            xi_x = self.attribute_interdependence(x=x, w=w_chunks, kappa_x=kappa_x, device=device)
            return xi_x
        else:
            return kappa_x if kappa_x is not None else x

    def calculate_instance_xi_x(self, x: torch.Tensor, channel_index: int = 0, kappa_x: torch.Tensor = None, device='cpu', *args, **kwargs):
        if self.instance_interdependence is not None:
            if self.instance_interdependence.device != device:
                self.instance_interdependence.to(device)

            if self.w_instance_interdependence is not None and 0 <= channel_index < self.w_instance_interdependence.size(0):
                w_chunks = self.w_instance_interdependence[channel_index:channel_index+1, :]
            else:
                w_chunks = None
            xi_x = self.instance_interdependence(x=x, w=w_chunks, kappa_x=kappa_x, device=device)
            return xi_x
        else:
            return kappa_x if kappa_x is not None else x

    # this function checks conditions for faster calculation across multi-channels...
    def calculate_kappa_xi_x(self, x: torch.Tensor, channel_index: int = 0, device='cpu', *args, **kwargs):

        # ************** Attribute Interdependence Block **************
        xi_x = self.calculate_attribute_xi_x(x=x, channel_index=channel_index, device=self.device)

        # ************** Data Expansion Block **************
        kappa_x = self.calculate_kappa_x(x=xi_x, device=device, *args, **kwargs)
        assert kappa_x.shape[1] == self.data_transformation.calculate_D(m=xi_x.shape[1])

        # ************** Instance Interdependence Block **************
        kappa_xi_x = self.calculate_instance_xi_x(x=x, channel_index=channel_index, kappa_x=kappa_x, device=self.device)

        return kappa_xi_x

    def calculate_inner_product(self, kappa_xi_x: torch.Tensor, phi_w: torch.Tensor, device: str = 'cpu', *args, **kwargs):
        if phi_w is not None:
            assert kappa_xi_x.ndim == 2 and phi_w.ndim == 2 and kappa_xi_x.size(-1) == phi_w.size(-1)
            if device != 'mps' and (kappa_xi_x.is_sparse or phi_w.is_sparse):
                inner_prod = torch.sparse.mm(kappa_xi_x, phi_w.T)
                if self.b is not None:
                    inner_prod += self.b
            else:
                inner_prod = F.linear(kappa_xi_x, phi_w, bias=self.b)
        else:
            inner_prod = kappa_xi_x
        return inner_prod

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
        # ************** Input Processing Block **************
        if x is None:
            raise ValueError("x cannot be None...")

        x = function.func_x(x=x, functions=self.input_process_functions, device=device)

        inner_products = []

        pre_computed_kappa_xi_x = None
        # if the instance functions has no parameters, it can be pre-computed and reused across channels
        if ((self.attribute_interdependence is None or not self.attribute_interdependence.require_parameters) and
            (self.instance_interdependence is None or not self.instance_interdependence.require_parameters)):
            pre_computed_kappa_xi_x = self.calculate_kappa_xi_x(x=x, channel_index=0, device=device)

        for channel_index in range(self.channel_num):

            # ************** Data Transformation Block **************
            if ((self.attribute_interdependence is None or not self.attribute_interdependence.require_parameters)
                and (self.instance_interdependence is None or not self.instance_interdependence.require_parameters)
                and pre_computed_kappa_xi_x is not None):
                kappa_xi_x = pre_computed_kappa_xi_x
            else:
                kappa_xi_x = self.calculate_kappa_xi_x(x=x, channel_index=channel_index, device=device)

            # ************** Parameter Reconciliation Block **************
            phi_w = self.calculate_phi_w(D=kappa_xi_x.size(1), channel_index=channel_index, device=device, *args, **kwargs)

            # ************** Inner Product Calculation Block **************
            inner_prod = self.calculate_inner_product(kappa_xi_x=kappa_xi_x, phi_w=phi_w, device=device, *args, **kwargs)
            inner_products.append(inner_prod)

        # ************** Multi-Channel Fusion Block **************
        if self.channel_fusion is not None:
            assert self.channel_fusion.get_dims() is None or self.channel_fusion.get_num() == len(inner_products)
            result = self.channel_fusion(x=inner_products, w=self.w_channel_fusion, device=device)
            n = self.channel_fusion.calculate_n(dims=[result.size(-1) for result in inner_products])
        else:
            assert len(inner_products) == 1
            result = inner_products[0]
            n = self.n
        assert result.size(-1) == n

        # ************** Remainder Block **************
        pi_x = self.calculate_pi_x(x=x, device=device, *args, **kwargs)
        if pi_x is not None:
            assert pi_x.size(-1) == n
            result += pi_x

        # ************** Output Processing Block **************
        return function.func_x(x=result, functions=self.output_process_functions, device=self.device)

