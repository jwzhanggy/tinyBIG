# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################
# Graph based RPN Layer Module #
################################

from tinybig.module.base_layer import rpn_layer
from tinybig.head.graph_based_heads import sgc_head


class sgc_layer(rpn_layer):
    def __init__(
        self,
        m: int, n: int,
        name: str = 'sgc_layer',
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
        heads = [
            sgc_head(
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
                with_lorr=with_lorr,
                r=r,
                with_residual=with_residual,
                enable_bias=enable_bias,
                device=device,
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class gat_layer(rpn_layer):
    def __init__(
        self,
        m: int, n: int,
        name: str = 'gat_layer',
        graph_file_path: str = None,
        nodes: list = None,
        links: list = None,
        directed: bool = True,
        self_dependence: bool = False,
        channel_num: int = 1,
        with_lorr: bool = False, r: int = 3,
        with_residual: bool = False,
        enable_bias: bool = False,
        width: int = 1,
        device: str = 'cpu', *args, **kwargs
    ):
        heads = [
            sgc_head(
                m=m, n=n,
                graph_file_path=graph_file_path,
                nodes=nodes,
                links=links,
                directed=directed,
                self_dependence=self_dependence,
                channel_num=channel_num,
                with_lorr=with_lorr,
                r=r,
                with_residual=with_residual,
                enable_bias=enable_bias,
                device=device,
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)