# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

############################
# Graph Based Head Modules #
############################

import pickle
import torch

from tinybig.module.base_head import rpn_head
from tinybig.interdependence.topological_interdependence import graph_interdependence, pagerank_multihop_graph_interdependence, multihop_graph_interdependence
from tinybig.interdependence.parameterized_bilinear_interdependence import lowrank_parameterized_bilinear_interdependence
from tinybig.interdependence.hybrid_interdependence import hybrid_interdependence
from tinybig.expansion.basic_expansion import identity_expansion
from tinybig.reconciliation.basic_reconciliation import identity_reconciliation
from tinybig.reconciliation.lowrank_reconciliation import lorr_reconciliation, dual_lphm_reconciliation
from tinybig.remainder.basic_remainder import zero_remainder, linear_remainder
from tinybig.fusion.metric_fusion import prod_fusion
from tinybig.koala.topology import graph as graph_class


class sgc_head(rpn_head):
    def __init__(
        self,
        m: int, n: int,
        name: str = 'sgc_head',
        channel_num: int = 1,
        # graph structure parameters
        graph: graph_class = None,
        graph_file_path: str = None,
        nodes: list = None,
        links: list = None,
        directed: bool = False,
        # graph interdependence function parameters
        with_multihop: bool = False, h: int = 1, accumulative: bool = False,
        with_pagerank: bool = False, c: float = 0.15,
        require_data: bool = False,
        require_parameters: bool = False,
        # adj matrix processing parameters
        normalization: bool = True,
        normalization_mode: str = 'row_column',
        self_dependence: bool = True,
        # parameter reconciliation and remainder functions
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        with_residual: bool = False,
        enable_bias: bool = False,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if graph is not None:
            graph_structure = graph
        elif graph_file_path is not None:
            graph_structure = graph_class.load(complete_path=graph_file_path)
        elif nodes is not None and links is not None:
            graph_structure = graph_class(
                nodes=nodes,
                links=links,
                directed=directed,
                device=device,
            )
        else:
            raise ValueError('You must provide a graph_file_path or nodes or links...')

        if with_pagerank:
            instance_interdependence = pagerank_multihop_graph_interdependence(
                b=graph_structure.get_node_num(), m=m,
                c=c,
                interdependence_type='instance',
                graph=graph_structure,
                normalization=normalization,
                normalization_mode=normalization_mode,
                self_dependence=self_dependence,
                require_data=require_data,
                require_parameters=require_parameters,
                device=device
            )
        elif with_multihop:
            instance_interdependence = multihop_graph_interdependence(
                b=graph_structure.get_node_num(), m=m,
                h=h, accumulative=accumulative,
                interdependence_type='instance',
                graph=graph_structure,
                normalization=normalization,
                normalization_mode=normalization_mode,
                self_dependence=self_dependence,
                require_data=require_data,
                require_parameters=require_parameters,
                device=device
            )
        else:
            instance_interdependence = graph_interdependence(
                b=graph_structure.get_node_num(), m=m,
                interdependence_type='instance',
                graph=graph_structure,
                normalization=normalization,
                normalization_mode=normalization_mode,
                self_dependence=self_dependence,
                require_data=require_data,
                require_parameters=require_parameters,
                device=device
            )

        data_transformation = identity_expansion(
            device=device
        )

        if with_dual_lphm:
            parameter_fabrication = dual_lphm_reconciliation(
                r=r,
                device=device,
                enable_bias=enable_bias,
            )
        elif with_lorr:
            parameter_fabrication = lorr_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device
            )
        else:
            parameter_fabrication = identity_reconciliation(
                enable_bias=enable_bias,
                device=device
            )

        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )

        super().__init__(
            m=m, n=n, name=name,
            instance_interdependence=instance_interdependence,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            channel_num=channel_num,
            device=device, *args, **kwargs
        )


class gat_head(sgc_head):
    def __init__(
        self,
        m: int, n: int,
        name: str = 'gat_head',
        channel_num: int = 1,
        # graph structure parameters
        graph: graph_class = None,
        graph_file_path: str = None,
        nodes: list = None,
        links: list = None,
        directed: bool = True,
        # graph interdependence function
        with_multihop: bool = False, h: int = 1, accumulative: bool = False,
        with_pagerank: bool = False, c: float = 0.15,
        require_data: bool = False,
        require_parameters: bool = False,
        # adj matrix processing parameters
        normalization: bool = True,
        normalization_mode: str = 'row_column',
        self_dependence: bool = False,
        # parameter reconciliation and remainder functions
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        with_residual: bool = False,
        enable_bias: bool = True,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if graph is not None:
            graph_structure = graph
        elif graph_file_path is not None:
            graph_structure = graph_class.load(complete_path=graph_file_path)
        elif nodes is not None and links is not None:
            graph_structure = graph_class(
                nodes=nodes,
                links=links,
                directed=directed,
                device=device,
            )
        else:
            raise ValueError('You must provide a graph_file_path or nodes or links...')

        graph_structure_interdependence = graph_interdependence(
            b=graph_structure.get_node_num(), m=m,
            graph=graph_structure,
            normalization=False,
            self_dependence=self_dependence,
            require_data=False,
            require_parameters=False,
            device=device
        )

        bilinear_interdependence = lowrank_parameterized_bilinear_interdependence(
            r=r,
            require_data=True,
            require_parameters=True,
            postprocess_functions=[torch.nn.Softmax(dim=0)],
            device=device,
        )

        if with_pagerank:
            graph_structure_interdependence = pagerank_multihop_graph_interdependence(
                b=graph_structure.get_node_num(), m=m,
                c=c,
                interdependence_type='instance',
                graph=graph_structure,
                normalization=normalization,
                normalization_mode=normalization_mode,
                self_dependence=self_dependence,
                require_data=require_data,
                require_parameters=require_parameters,
                device=device
            )
        elif with_multihop:
            graph_structure_interdependence = multihop_graph_interdependence(
                b=graph_structure.get_node_num(), m=m,
                h=h, accumulative=accumulative,
                interdependence_type='instance',
                graph=graph_structure,
                normalization=normalization,
                normalization_mode=normalization_mode,
                self_dependence=self_dependence,
                require_data=require_data,
                require_parameters=require_parameters,
                device=device
            )
        else:
            graph_structure_interdependence = graph_interdependence(
                b=graph_structure.get_node_num(), m=m,
                interdependence_type='instance',
                graph=graph_structure,
                normalization=normalization,
                normalization_mode=normalization_mode,
                self_dependence=self_dependence,
                require_data=require_data,
                require_parameters=require_parameters,
                device=device
            )

        instance_interdependence = hybrid_interdependence(
            b=graph_structure.get_node_num(), m=m,
            interdependence_functions=[
                graph_structure_interdependence,
                bilinear_interdependence,
            ],
            fusion_function=prod_fusion,
        )

        data_transformation = identity_expansion(
            device=device
        )

        if with_dual_lphm:
            parameter_fabrication = dual_lphm_reconciliation(
                r=r,
                device=device,
                enable_bias=enable_bias,
            )
        elif with_lorr:
            parameter_fabrication = lorr_reconciliation(
                r=r,
                enable_bias=enable_bias,
                device=device
            )
        else:
            parameter_fabrication = identity_reconciliation(
                enable_bias=enable_bias,
                device=device
            )

        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )

        super().__init__(
            m=m, n=n, name=name,
            instance_interdependence=instance_interdependence,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            channel_num=channel_num,
            device=device, *args, **kwargs
        )
