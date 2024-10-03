# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based GCN Model #
#######################

from tinybig.model.rpn import rpn
from tinybig.layer.graph_based_layers import sgc_layer


class gcn(rpn):
    def __init__(
        self,
        dims: list[int] | tuple[int],
        name: str = 'rpn_gcn',
        graph_file_path: str = None,
        nodes: list = None,
        links: list = None,
        directed: bool = True,
        normalization: bool = False,
        normalization_mode: str = 'row_column',
        self_dependence: bool = False,
        require_data: bool = False,
        require_parameters: bool = False,
        channel_num: int = 1,
        with_lorr: bool = False, r: int = 3,
        with_residual: bool = False,
        enable_bias: bool = False,
        width: int = 1,
        device: str = 'cpu', *args, **kwargs
    ):
        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            layers.append(
                sgc_layer(
                    m=m, n=n,
                    graph_file_path=graph_file_path,
                    nodes=nodes,
                    links=links,
                    directed=directed,
                    normalization=normalization,
                    normalization_mode=normalization_mode,
                    self_dependence=self_dependence,
                    require_data=require_data,
                    require_parameters=require_parameters,
                    channel_num=channel_num,
                    with_lorr=with_lorr, r=r,
                    with_residual=with_residual,
                    enable_bias=enable_bias,
                    width=width,
                    device=device,
                )
            )
        super().__init__(name=name, layers=layers, device=device, *args, **kwargs)

