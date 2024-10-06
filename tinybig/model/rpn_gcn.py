# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based GCN Model #
#######################

from tinybig.model.rpn import rpn
from tinybig.layer.graph_based_layers import sgc_layer
from tinybig.koala.topology import graph as graph_class


class gcn(rpn):
    def __init__(
        self,
        dims: list[int] | tuple[int],
        channel_num: int = 1,
        width: int = 1,
        name: str = 'rpn_gcn',
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
        with_dropout: bool = True, p: float = 0.25,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                sgc_layer(
                    m=m, n=n,
                    width=width,
                    channel_num=channel_num,
                    # ---------------
                    graph=graph,
                    graph_file_path=graph_file_path,
                    nodes=nodes,
                    links=links,
                    directed=directed,
                    # ---------------
                    with_multihop=with_multihop, h=h, accumulative=accumulative,
                    with_pagerank=with_pagerank, c=c,
                    require_data=require_data,
                    require_parameters=require_parameters,
                    # ---------------
                    normalization=normalization,
                    normalization_mode=normalization_mode,
                    self_dependence=self_dependence,
                    # ---------------
                    with_dual_lphm=with_dual_lphm,
                    with_lorr=with_lorr, r=r,
                    with_residual=with_residual,
                    enable_bias=enable_bias,
                    # ---------------
                    with_batch_norm=with_batch_norm and n != dims[-1],
                    with_relu=with_relu and n != dims[-1],
                    with_dropout=with_dropout and n != dims[-1], p=p,
                    with_softmax=with_softmax and m == dims[-2] and n == dims[-1],
                    # ---------------
                    device=device,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

