# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################
# Graph based RPN Layer Module #
################################

from tinybig.module.base_layer import rpn_layer
from tinybig.head.graph_based_heads import sgc_head, gat_head
from tinybig.koala.topology import graph as graph_class


class sgc_layer(rpn_layer):
    def __init__(
        self,
        m: int, n: int,
        width: int = 1,
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
        # output processing parameters
        with_batch_norm: bool = False,
        with_relu: bool = True,
        with_softmax: bool = True,
        with_dropout: bool = True, p: float = 0.5,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            sgc_head(
                m=m, n=n,
                channel_num=channel_num,
                # -------------------
                graph=graph,
                graph_file_path=graph_file_path,
                nodes=nodes,
                links=links,
                directed=directed,
                # -------------------
                with_multihop=with_multihop, h=h, accumulative=accumulative,
                with_pagerank=with_pagerank, c=c,
                require_data=require_data,
                require_parameters=require_parameters,
                # -------------------
                normalization=normalization,
                normalization_mode=normalization_mode,
                self_dependence=self_dependence,
                # -------------------
                with_dual_lphm=with_dual_lphm,
                with_lorr=with_lorr, r=r,
                with_residual=with_residual,
                enable_bias=enable_bias,
                # -------------------
                with_batch_norm=with_batch_norm,
                with_relu=with_relu,
                with_softmax=with_softmax,
                with_dropout=with_dropout, p=p,
                # -------------------
                device=device,
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class gat_layer(rpn_layer):
    def __init__(
        self,
        m: int, n: int,
        width: int = 1,
        name: str = 'gat_layer',
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
        normalization: bool = False,
        normalization_mode: str = 'row_column',
        self_dependence: bool = True,
        # bilinear interdependence function parameters
        with_dual_lphm_interdependence: bool = False,
        with_lorr_interdependence: bool = False, r_interdependence: int = 3,
        # parameter reconciliation and remainder functions
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        with_residual: bool = False,
        enable_bias: bool = False,
        # output processing parameters
        with_batch_norm: bool = False,
        with_relu: bool = True,
        with_softmax: bool = True,
        with_dropout: bool = True, p: float = 0.5,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            gat_head(
                m=m, n=n,
                channel_num=channel_num,
                # -------------------
                graph=graph,
                graph_file_path=graph_file_path,
                nodes=nodes,
                links=links,
                directed=directed,
                # -------------------
                with_multihop=with_multihop, h=h, accumulative=accumulative,
                with_pagerank=with_pagerank, c=c,
                require_data=require_data,
                require_parameters=require_parameters,
                # -------------------
                normalization=normalization,
                normalization_mode=normalization_mode,
                self_dependence=self_dependence,
                # -------------------
                with_dual_lphm_interdependence=with_dual_lphm_interdependence,
                with_lorr_interdependence=with_lorr_interdependence, r_interdependence=r_interdependence,
                # -------------------
                with_dual_lphm=with_dual_lphm,
                with_lorr=with_lorr, r=r,
                with_residual=with_residual,
                enable_bias=enable_bias,
                # -------------------
                with_batch_norm=with_batch_norm,
                with_relu=with_relu,
                with_softmax=with_softmax,
                with_dropout=with_dropout, p=p,
                # -------------------
                device=device,
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)
