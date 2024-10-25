# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################################
# RPN Multi-Head Base Layer Module #
####################################

"""
The RPN layer with multi-heads.

This module contains the implementation of RPN layer with multiple RPN heads.
The RPN layer will be used to compose the deep RPN models by stacking them.
"""

import torch
import math

from tinybig.config import config
from tinybig.fusion.metric_fusion import mean_fusion
from tinybig.module.base_fusion import fusion


class rpn_layer(torch.nn.Module):
    r"""
    The RPN layer class for implementing the multi-head module.

    It will be used to compose the RPN model with deep architectures.

    ...

    Notes
    ----------
    Similar to the Transformers, for each layer of RPN model, it allows a multi-head architecture,
    where each head will disentangle the input data and model parameters using different expansion,
    reconciliation and remainder functions shown as follows:
    $$
        \begin{equation}
            g(\mathbf{x} | \mathbf{w}, H) = \sum_{h=0}^{H-1} \left\langle \kappa^{(h)}(\mathbf{x}), \psi^{(h)}(\mathbf{w}^{(h)}) \right\rangle + \pi^{(h)}(\mathbf{x}),
        \end{equation}
    $$
    where the superscript "$h$" indicates the head index and $H$ denotes the total head number.
    By default, summation is used to combine the results from all these heads.

    Attributes
    ----------
    m: int
        The input dimension of the layer.
    n: int
        The output dimension of the layer.
    heads: torch.nn.ModuleList, default = torch.nn.ModuleList()
        The list of RPN heads involved in the layer.
    head_fusion: fusion, default = None
        The fusion function of the outputs learned by multi-heads.
    device: str, default = 'cpu'
            The device for hosting the RPN layer.

    Methods
    ----------
    __init__
        The initialization method of the RPN-layer module with multiple RPN heads.

    get_widthber
        The head number retrieval method.

    initialize_parameters
        Head parameter initialization method.

    initialize_fusion_parameters
        Fusion component parameter initialization method.

    multi_head_fusion
        The multi-head outputs fusion method.

    forward
        The forward method of this multi-head PRN layer module.

    __call__
        The re-implementatino of the callable method of this RPN layer module.
    """
    def __init__(
        self,
        m: int,
        n: int,
        name: str = "rpn_layer",
        heads: list = None,
        head_configs: dict | list = None,
        width: int = None,
        width_alloc: int | list = None,
        head_fusion=None,
        head_fusion_configs=None,
        parameters_init_method: str = 'xavier_uniform',
        device='cpu',
        *args, **kwargs
    ):
        r"""
        The initialization method of the RPN-layer module with multiple RPN heads.

        It initializes the RPN layer module composed with multiple RPN heads.
        Specifically, this method initializes the dimension configurations of the layer,
        the component heads, and defines the device to host the head.

        Parameters
        ----------
        m: int
            The input dimension of the layer.
        n: int
            The output dimension of the layer.
        heads: torch.nn.ModuleList, default = torch.nn.ModuleList()
            The list of RPN heads involved in the layer. The heads involved in the layer can be initialized
            either directly with the heads parameter or via the head_configs parameter.
        head_configs: list, default = None
            The list of RPN head configurations in the layer.
        width: int, default = None
            The total head number of the layer. It is optional, if the "heads" or the "head_configs" can provide
            sufficient information for the head initialization, this widthber parameter can be set as None.
        width_alloc: list, default = None
            RPN allows the heads with different configurations, instead of listing such configurations one by one,
            it also allows the listing of each configuration types together with the repeating numbers for
            each of them, which are specified by this optional head number allocation parameter.
        head_fusion: fusion, default = None
            The fusion function of the outputs learned by multi-heads.
        head_fusion_configs: dict, default = None
            The fusion function configurations of the outputs learned by multi-heads.
        device: str, default = 'cpu'
            The device for hosting the RPN layer.

        Returns
        ----------
        object
            This method will return the initialized RPN-layer object.
        """
        super().__init__()
        assert m is not None and n is not None
        self.m = m
        self.n = n
        self.name = name
        self.fusion_parameters = None
        self.heads = torch.nn.ModuleList()
        self.parameters_init_method = parameters_init_method
        self.device = device

        # the multi-head initialization
        if heads is not None:
            # initialize heads from the provided head parameter directly
            self.heads.extend(heads)
            width = len(self.heads)
        elif head_configs is None:
            raise ValueError("Both heads and head_configs are None, this layer cannot be initialized...")
        else:
            # initialize heads from the head configs

            # process the width, width_alloc and head_configs parameters to make them consistent
            width, width_alloc, head_configs = config.process_num_alloc_configs(width, width_alloc, head_configs)
            assert len(width_alloc) == len(head_configs) and sum(width_alloc) == width

            # initialize the multi-head involved in the layer
            for head_repeat_time, head_config in zip(width_alloc, head_configs):
                for head_id in range(0, head_repeat_time):
                    head_class_name = head_config['head_class']
                    head_parameters = head_config['head_parameters']
                    head_parameters['m'] = self.m
                    head_parameters['n'] = self.n
                    head_parameters['device'] = device
                    head_parameters['parameters_init_method'] = self.parameters_init_method
                    self.heads.append(config.get_obj_from_str(head_class_name)(**head_parameters))

        assert len(self.heads) == width and [(self.m, self.n)] * width == [(head.m, head.n) for head in self.heads]

        self.head_fusion = config.instantiation_functions(functions=head_fusion, function_configs=head_fusion_configs, device=device)
        if len(self.heads) > 1 and self.head_fusion is None:
            self.head_fusion = mean_fusion(dims=[head.get_n() for head in heads])
        self.w_head_fusion = None
        self.create_learnable_parameters(init_type=self.parameters_init_method)

    def get_m(self):
        return self.m

    def get_n(self):
        if self.head_fusion is not None:
            return self.head_fusion.calculate_n()
        else:
            return self.n

    def create_learnable_parameters(self, initialize_parameter_at_creation: bool = False, init_type='xavier_uniform', init_bias=True, *args, **kwargs):
        if self.head_fusion is not None and self.head_fusion.require_parameters:
            l = self.head_fusion.calculate_l()
            self.w_head_fusion = torch.nn.Parameter(torch.rand(1, l))

        if initialize_parameter_at_creation:
            self.initialize_parameters(init_type=init_type, init_bias=init_bias)

    def initialize_parameters(self, init_type='xavier_uniform', init_bias=True):
        """
        Head parameter initialization method.

        It initializes the learnable parameters in each head involved in the layer,
        which will call the parameter initialization method in each of the heads.

        Returns
        -------
        None
            The initialization method doesn't have any return values.
        """
        for head in self.heads:
            head.initialize_parameters(init_type=init_type, init_bias=init_bias)
        if self.w_head_fusion is not None:
            if init_type == 'xavier_uniform':
                torch.nn.init.kaiming_uniform_(self.w_head_fusion, a=math.sqrt(5))
            else:
                torch.nn.init.xavier_uniform_(self.w_head_fusion)

    def initialize_fusion_parameters(self):
        """
        Fusion component parameter initialization method.

        It initializes the learnable parameters for the fusion component.
        The RPN head also allows the linear fusion component to combine the
        outputs of multi-head with learnable parameters.

        Returns
        -------
        None
            The initialization method doesn't have any return values.
        """
        self.fusion_parameters = torch.nn.Parameter(torch.rand(self.n, self.n*len(self.heads)))
        torch.nn.init.xavier_uniform_(self.fusion_parameters)

    def get_width(self):
        """
        The head number retrieval method.

        It returns the head number of the layer.

        Returns
        -------
        int
            The number of heads in the layer.
        """
        return len(self.heads)

    def to_config(self):
        layer_class = f"{self.__class__.__module__}.{self.__class__.__name__}"
        layer_parameters = {
            'name': self.name,
            'device': self.device,
            'm': self.m,
            'n': self.n,
            'head_configs': [head.to_config() for head in self.heads] if self.head else [],
        }

        if self.head_fusion is not None:
            layer_parameters['head_fusion_configs']= self.head_fusion.to_config()

        return {
            "layer_class": layer_class,
            "layer_parameters": layer_parameters
        }

    def __call__(self, *args, **kwargs):
        """
        The re-implementatino of the callable method of this RPN layer module.

        It re-implements the builtin callable method by calling the forward method.

        Returns
        -------
        torch.Tensor
            It will return the learning results of this RPN layer.
        """
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor, fusion_strategy: str = 'average', device: str = 'cpu', *args, **kwargs):
        r"""
        The forward method of this multi-head PRN layer module.

        It calculates the outputs with the multi-head RPN layer based on the inputs subject to certain fusion strategy.
        For each layer of RPN model, RPN allows a multi-head architecture,
        where each head will disentangle the input data and model parameters using different expansion,
        reconciliation and remainder functions shown as follows:
        $$
            \begin{equation}
                g(\mathbf{x} | \mathbf{w}, H) = \sum_{h=0}^{H-1} \left\langle \kappa^{(h)}(\mathbf{x}), \psi^{(h)}(\mathbf{w}^{(h)}) \right\rangle + \pi^{(h)}(\mathbf{x}),
            \end{equation}
        $$
        where the superscript "$h$" indicates the head index and $H$ denotes the total head number.
        By default, summation is used to combine the results from all these heads.

        Parameters
        ----------
        x: torch.Tensor
            The input data to the layer.
        fusion_strategy: str, default = 'average'
            The optional fusion_strategy of the forward method. If it is set as None, this layer will use the default
             fusion_strategy at initialization of this layer.
        device: str, default = 'cpu'
            Device used to host this layer for calculation.

        Returns
        -------
        torch.Tensor
            It will return the learning results of this RPN layer.
        """
        assert x is not None and x.ndim == 2 and x.size(1) == self.get_m()

        results = []
        for head in self.heads:
            results.append(head(x=x, device=device))
        assert results != [] and [results[0].shape] * len(results) == [result.shape for result in results]

        if self.head_fusion is not None:
            assert self.head_fusion.get_num() == len(results) and [results[0].shape] * len(results) == [result.shape for result in results]
            result = self.head_fusion(x=results, w=self.w_head_fusion, device=device)
        else:
            assert len(results) == 1
            result = results[0]

        assert result.size(1) == self.get_n()
        return result