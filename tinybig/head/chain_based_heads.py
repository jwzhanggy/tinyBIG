# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################
# Chain Based Head Modules #
###########################

"""
Chain Structural RPN based heads.

This module contains the chain structural rpn based heads, including
    graph_interdependence_head

"""

import torch
import torch.nn.functional as F

from tinybig.module.base_head import head
from tinybig.remainder import zero_remainder
from tinybig.koala.topology import chain
from tinybig.interdependence import (
    chain_interdependence,
    multihop_chain_interdependence,
    inverse_approx_multihop_chain_interdependence,
    exponential_approx_multihop_chain_interdependence
)
from tinybig.reconciliation import (
    identity_reconciliation,
    lorr_reconciliation,
    dual_lphm_reconciliation
)
from tinybig.expansion import (
    identity_expansion,
)
from tinybig.remainder import (
    zero_remainder,
    linear_remainder
)


class chain_interdependence_head(head):
    """
    A head class that implements chain-based interdependence mechanisms for multi-channel modules.

    This head supports chain-based instance interdependence, various data transformations, parameter reconciliation,
    and customizable output processing functions.

    Attributes
    ----------
    m : int
        Input dimension of the head.
    n : int
        Output dimension of the head.
    chain_length : int
        Length of the chain structure used for interdependence.
    channel_num : int
        Number of channels for multi-channel processing.
    name : str
        Name of the head.
    parameters_init_method : str
        Initialization method for parameters.
    device : str
        Device to host the head (e.g., 'cpu' or 'cuda').
    """
    def __init__(
        self,
        m: int, n: int,
        chain_length: int,
        channel_num: int = 1,
        name: str = 'chain_interdependence_head',
        # interdependence function parameters
        bi_directional: bool = False,
        with_multihop: bool = False, h: int = 1, accumulative: bool = False,
        with_inverse_approx: bool = False,
        with_exponential_approx: bool = False,
        self_dependence: bool = True,
        self_scaling: float = 1.0,
        # data transformation function parameters
        with_taylor: bool = False, d: int = 2,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = False,
        # remainder function parameters
        with_residual: bool = False,
        # output processing parameters
        with_batch_norm: bool = False,
        with_relu: bool = True,
        with_softmax: bool = True,
        with_dropout: bool = False, p: float = 0.25,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize a chain-based interdependence head.

        Parameters
        ----------
        m : int
            Input dimension of the head.
        n : int
            Output dimension of the head.
        chain_length : int
            Length of the chain structure used for interdependence.
        channel_num : int, optional
            Number of channels for multi-channel processing, default is 1.
        name : str, optional
            Name of the head, default is 'chain_interdependence_head'.
        bi_directional : bool, optional
            Whether the chain is bi-directional, default is False.
        with_multihop : bool, optional
            Whether to enable multi-hop connections, default is False.
        h : int, optional
            Number of hops for multi-hop connections, default is 1.
        accumulative : bool, optional
            Whether the multi-hop connections are accumulative, default is False.
        with_inverse_approx : bool, optional
            Whether to use inverse approximation for chain interdependence, default is False.
        with_exponential_approx : bool, optional
            Whether to use exponential approximation for chain interdependence, default is False.
        self_dependence : bool, optional
            Whether to include self-dependence in the chain, default is True.
        self_scaling : float, optional
            Scaling factor for self-dependence, default is 1.0.
        with_taylor : bool, optional
            Whether to use Taylor expansion for data transformation, default is False.
        d : int, optional
            Degree of Taylor expansion, default is 2.
        with_dual_lphm : bool, optional
            Whether to use dual LPHM for parameter reconciliation, default is False.
        with_lorr : bool, optional
            Whether to use LORR for parameter reconciliation, default is False.
        r : int, optional
            Rank for parameter reconciliation, default is 3.
        enable_bias : bool, optional
            Whether to enable bias in parameter reconciliation, default is False.
        with_residual : bool, optional
            Whether to include a residual connection in the remainder function, default is False.
        with_batch_norm : bool, optional
            Whether to include batch normalization in output processing, default is False.
        with_relu : bool, optional
            Whether to include ReLU activation in output processing, default is True.
        with_softmax : bool, optional
            Whether to include softmax activation in output processing, default is True.
        with_dropout : bool, optional
            Whether to include dropout in output processing, default is False.
        p : float, optional
            Dropout probability, default is 0.25.
        parameters_init_method : str, optional
            Initialization method for parameters, default is 'xavier_normal'.
        device : str, optional
            Device to host the head, default is 'cpu'.

        Returns
        -------
        None
        """
        self.chain_length = chain_length
        chain_structure = chain(
            length=chain_length,
            name=name,
            bi_directional=bi_directional,
            device=device,
        )

        if with_exponential_approx:
            instance_interdependence = exponential_approx_multihop_chain_interdependence(
                b=chain_length, m=m,
                chain=chain_structure,
                interdependence_type='instance',
                normalization=False,
                self_dependence=self_dependence,
                self_scaling=self_scaling,
                require_data=False,
                require_parameters=False,
            )
        elif with_inverse_approx:
            instance_interdependence = inverse_approx_multihop_chain_interdependence(
                b=chain_length, m=m,
                chain=chain_structure,
                interdependence_type='instance',
                normalization=False,
                self_dependence=self_dependence,
                self_scaling=self_scaling,
                require_data=False,
                require_parameters=False,
            )
        elif with_multihop:
            instance_interdependence = multihop_chain_interdependence(
                h=h, accumulative=accumulative,
                b=chain_length, m=m,
                chain=chain_structure,
                interdependence_type='instance',
                normalization=False,
                self_dependence=self_dependence,
                self_scaling=self_scaling,
                require_data=False,
                require_parameters=False,
            )
        else:
            instance_interdependence = chain_interdependence(
                b=chain_length, m=m,
                chain=chain_structure,
                interdependence_type='instance',
                normalization=False,
                self_dependence=self_dependence,
                self_scaling=self_scaling,
                require_data=False,
                require_parameters=False,
            )
        print('** instance_interdependence', instance_interdependence)
        print('*** bi_directional', bi_directional)

        data_transformation = identity_expansion(
            device=device,
        )
        print('** data_transformation', data_transformation)

        if with_dual_lphm:
            parameter_fabrication = dual_lphm_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device
            )
        elif with_lorr:
            parameter_fabrication = lorr_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device,
            )
        else:
            parameter_fabrication = identity_reconciliation(
                enable_bias=enable_bias,
                device=device,
            )
        l = parameter_fabrication.calculate_l(
            n=n, D=data_transformation.calculate_D(m=m)
        )
        print('** parameter_fabrication', parameter_fabrication)

        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )
        print('** remainder', remainder)

        output_process_functions = []
        if with_batch_norm:
            output_process_functions.append(torch.nn.BatchNorm1d(num_features=n, device=device))
        if with_relu:
            output_process_functions.append(torch.nn.ReLU())
        if with_dropout:
            output_process_functions.append(torch.nn.Dropout(p=p))
        if with_softmax:
            output_process_functions.append(torch.nn.Softmax(dim=-1))
        print('** output_process_functions', output_process_functions)

        super().__init__(
            m=m, n=n,
            name=name, channel_num=channel_num,
            batch_num=chain_length,
            instance_interdependence=instance_interdependence,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            parameters_init_method=parameters_init_method,
            device=device, *args, **kwargs
        )

    def calculate_instance_xi_x(self, x: torch.Tensor, channel_index: int = 0, kappa_x: torch.Tensor = None, device='cpu', *args, **kwargs):
        """
        Calculate the instance-based interdependence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        channel_index : int, optional
            Index of the channel for multi-channel processing, default is 0.
        kappa_x : torch.Tensor, optional
            Pre-computed transformation of the input, default is None.
        device : str, optional
            Device to host the computation, default is 'cpu'.

        Returns
        -------
        torch.Tensor
            Transformed tensor with interdependence applied.
        """
        if self.instance_interdependence is not None:
            if self.instance_interdependence.device != device:
                self.instance_interdependence.to(device)

            if self.w_instance_interdependence is not None and 0 <= channel_index < self.w_instance_interdependence.size(0):
                w_chunks = self.w_instance_interdependence[channel_index:channel_index+1, :]
            else:
                w_chunks = None
            b, m = kappa_x.shape
            kappa_x = kappa_x.view(b, self.chain_length, -1).permute(1, 0, 2).reshape(self.chain_length, -1)
            xi_x = self.instance_interdependence(x=x, w=w_chunks, kappa_x=kappa_x, device=device)
            xi_x = xi_x.view(self.chain_length, b, -1).permute(1, 0, 2)
            return xi_x
        else:
            return kappa_x if kappa_x is not None else x

    def calculate_inner_product(self, kappa_xi_x: torch.Tensor, phi_w: torch.Tensor, device: str = 'cpu', *args, **kwargs):
        """
        Calculate the inner product of transformed data and reconciled parameters.

        Parameters
        ----------
        kappa_xi_x : torch.Tensor
            Transformed input data.
        phi_w : torch.Tensor
            Reconciled parameter tensor.
        device : str, optional
            Device to host the computation, default is 'cpu'.

        Returns
        -------
        torch.Tensor
            Resulting inner product tensor.
        """
        if phi_w is not None:
            assert phi_w.ndim == 2 and kappa_xi_x.size(-1) == phi_w.size(-1)
            if device != 'mps' and (kappa_xi_x.is_sparse or phi_w.is_sparse):
                inner_prod = torch.sparse.mm(kappa_xi_x, phi_w.T)
                if self.b is not None:
                    inner_prod += self.b
            else:
                inner_prod = F.linear(kappa_xi_x, phi_w, bias=self.b)
            inner_prod = inner_prod.view(inner_prod.size(0), -1)
        else:
            inner_prod = kappa_xi_x
        return inner_prod

    def fusion(self, inner_products: list[torch.Tensor], device: str = 'cpu', *args, **kwargs):
        """
        Fuse multi-channel inner products.

        Parameters
        ----------
        inner_products : list[torch.Tensor]
            List of inner product tensors for each channel.
        device : str, optional
            Device to host the computation, default is 'cpu'.

        Returns
        -------
        torch.Tensor
            Fused tensor after combining all channels.
        """
        if self.channel_fusion is not None:
            assert self.channel_fusion.get_dims() is None or self.channel_fusion.get_num() == len(inner_products)
            result = self.channel_fusion(x=inner_products, w=self.w_channel_fusion, device=device)
            n = self.channel_fusion.calculate_n(dims=[result.size(-1) for result in inner_products])
        else:
            assert len(inner_products) == 1
            result = inner_products[0]
            n = self.n
        assert result.size(-1) == n*self.chain_length
        return result

    def calculate_pi_x(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
        Calculate the remainder function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        device : str, optional
            Device to host the computation, default is 'cpu'.

        Returns
        -------
        torch.Tensor
            Remainder component, or None if not applicable.
        """
        if self.remainder is not None:
            if isinstance(self.remainder, zero_remainder):
                return None
            if self.remainder.device != device:
                self.remainder.to(device)
            b, m = x.shape
            x = x.view(b * self.chain_length, -1)
            pi_x = self.remainder(x=x, w=self.w_remainder, b=self.b_remainder, m=self.m, n=self.n, device=device)
            pi_x = pi_x.view(b, -1)
            return pi_x
        else:
            return None


