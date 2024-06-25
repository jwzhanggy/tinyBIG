# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
import math
import torch
import torch.nn.functional as F

from tinybig.util import get_obj_from_str, process_function_list, func_x, register_function_parameters


class rpn_head(torch.nn.Module):

    def __init__(self, m: int = None, n: int = None, l: int = None, channel_num: int = 1,
                 data_transformation=None, parameter_fabrication=None, remainder=None, output_process_functions=None,
                 data_transformation_class=None, parameter_fabrication_class=None, remainder_class=None,
                 output_process_function_configs=None, device='cpu', *args, **kwargs):
        super().__init__()
        assert channel_num >= 1 and m is not None and m >= 1 and n is not None and n >= 1

        self.m = m
        self.n = n
        self.channel_num = channel_num
        # l can be calculated with parameter_fabrication function, it can be none
        self.l = l
        self.device = device

        # data_transformation, parameter_fabrication and remainder initialize from either input objects or input configs
        if data_transformation is not None:
            self.data_transformation = data_transformation
        elif data_transformation_class is not None:
            parameters = kwargs['data_transformation_parameters'] if 'data_transformation_parameters' in kwargs else {}
            parameters['device'] = device
            self.data_transformation = get_obj_from_str(data_transformation_class)(**parameters)
        else:
            self.data_transformation = None

        if parameter_fabrication is not None:
            self.parameter_fabrication = parameter_fabrication
            assert self.parameter_fabrication.channel_number == self.channel_num
        elif parameter_fabrication_class is not None:
            parameters = kwargs['parameter_fabrication_parameters'] if 'parameter_fabrication_parameters' in kwargs else {}
            parameters['device'] = device
            self.parameter_fabrication = get_obj_from_str(parameter_fabrication_class)(**parameters)
        else:
            self.parameter_fabrication = None

        if remainder is not None:
            self.remainder = remainder
        elif remainder_class is not None:
            parameters = kwargs['remainder_parameters'] if 'remainder_parameters' in kwargs else {}
            parameters['device'] = device
            self.remainder = get_obj_from_str(remainder_class)(**parameters)
        else:
            self.remainder = None

        self.output_process_functions = process_function_list(output_process_functions, output_process_function_configs, device=device)
        #register_function_parameters(self, self.output_process_functions)

        # parameters for parameter_fabrication function
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

        # parameters for remainder function
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
        if init_type == 'kaiming_uniform':
            self.initialize_parameters_kaiming(init_bias=init_bias, *args, **kwargs)
        elif init_type == 'xavier_uniform':
            self.initialize_parameters_xavier(init_bias=init_bias, *args, **kwargs)

    def initialize_parameters_kaiming(self, init_bias=True, *args, **kwargs):
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
        if self.w is not None:
            torch.nn.init.xavier_uniform_(self.w)
        if self.w_prime is not None:
            torch.nn.init.xavier_uniform_(self.w_prime)
        if init_bias:
            if self.b is not None:
                torch.nn.init.xavier_uniform_(self.b.view(1, -1))
            if self.b_prime is not None:
                torch.nn.init.xavier_uniform_(self.b_prime.view(1, -1))

    def update_data_transformation(self, new_data_transformation):
        self.data_transformation = new_data_transformation

    def update_parameter_fabrication(self, new_parameter_fabrication):
        self.parameter_fabrication = new_parameter_fabrication

    def update_remainder(self, new_remainder):
        self.remainder = new_remainder

    def update_parameter_number(self, new_l=None):
        self.l = new_l
        self.w = torch.nn.Parameter(torch.rand(1, self.l))
        self.initialize_parameters(init_bias=False)

    def output_process(self, x: torch.Tensor, *args, **kwargs):
        return func_x(x, self.output_process_functions, device=self.device)

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
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
        if self.output_process_functions is not None:
            result = self.output_process(x=result)

        # [batch, n]
        return result

    def __call__(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        return self.forward(x=x, device=device, *args, **kwargs)
