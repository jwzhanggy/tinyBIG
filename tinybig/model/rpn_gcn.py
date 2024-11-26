# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#######################
# RPN based GCN Model #
#######################

"""
RPN based GNN models

This module contains the implementation of the RPN based GNN models, including
    gcn
"""

from tinybig.model.rpn import rpn
from tinybig.layer.graph_based_layers import graph_interdependence_layer
from tinybig.koala.topology import graph as graph_class


class gcn(rpn):

    """
    Graph Convolutional Network (GCN) model within the RPN framework.
    This class constructs a GCN architecture using a series of `graph_interdependence_layer` components.

    Parameters
    ----------
    dims : list[int] | tuple[int]
        List of layer dimensions for the GCN. Must contain at least two dimensions.
    channel_num : int, optional
        Number of channels for interdependence layers. Default is 1.
    width : int, optional
        Number of parallel heads in each layer. Default is 1.
    name : str, optional
        Name of the GCN model. Default is 'rpn_gcn'.
    graph : graph_class, optional
        Predefined graph structure. Default is None.
    graph_file_path : str, optional
        Path to a file defining the graph structure. Default is None.
    nodes : list, optional
        List of graph nodes. Required if `graph` or `graph_file_path` is not provided. Default is None.
    links : list, optional
        List of graph edges. Required if `graph` or `graph_file_path` is not provided. Default is None.
    directed : bool, optional
        Whether the graph is directed. Default is False.
    with_multihop : bool, optional
        Enables multihop interdependence in graph layers. Default is False.
    h : int, optional
        Number of hops for multihop interdependence. Default is 1.
    accumulative : bool, optional
        Accumulate interdependence over hops. Default is False.
    with_pagerank : bool, optional
        Include PageRank-based processing in layers. Default is False.
    c : float, optional
        Damping factor for PageRank. Default is 0.15.
    require_data : bool, optional
        Require data input for adjacency matrix processing. Default is False.
    require_parameters : bool, optional
        Require parameter input for adjacency matrix processing. Default is False.
    normalization : bool, optional
        Normalize adjacency matrix. Default is True.
    normalization_mode : str, optional
        Mode of normalization ('row' or 'column'). Default is 'column'.
    self_dependence : bool, optional
        Include self-loops in graph processing. Default is True.
    with_dual_lphm : bool, optional
        Enable dual LPHM parameter reconciliation. Default is False.
    with_lorr : bool, optional
        Enable LoRR parameter reconciliation. Default is False.
    r : int, optional
        Rank parameter for parameter reconciliation. Default is 3.
    with_residual : bool, optional
        Include residual connections in layers. Default is False.
    enable_bias : bool, optional
        Enable bias in parameter reconciliation. Default is False.
    with_batch_norm : bool, optional
        Include batch normalization in layers. Default is False.
    with_relu : bool, optional
        Include ReLU activation in layers. Default is True.
    with_softmax : bool, optional
        Include softmax activation in the output layer. Default is True.
    with_dropout : bool, optional
        Include dropout in layers. Default is False.
    p : float, optional
        Dropout probability. Default is 0.25.
    device : str, optional
        Device for computation ('cpu' or 'cuda'). Default is 'cpu'.
    *args : optional
        Additional positional arguments for superclass.
    **kwargs : optional
        Additional keyword arguments for superclass.

    Raises
    ------
    ValueError
        If `dims` contains fewer than two dimensions.

    """

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
        normalization_mode: str = 'column',
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
        with_dropout: bool = False, p: float = 0.25,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the GCN (Graph Convolutional Network) model.

        Parameters
        ----------
        dims : list[int] | tuple[int]
            List of layer dimensions for the GCN. Must contain at least two dimensions.
        channel_num : int, optional
            Number of channels for interdependence layers. Default is 1.
        width : int, optional
            Number of parallel heads in each layer. Default is 1.
        name : str, optional
            Name of the GCN model. Default is 'rpn_gcn'.
        graph : graph_class, optional
            Predefined graph structure. Default is None.
        graph_file_path : str, optional
            Path to a file defining the graph structure. Default is None.
        nodes : list, optional
            List of graph nodes. Required if `graph` or `graph_file_path` is not provided. Default is None.
        links : list, optional
            List of graph edges. Required if `graph` or `graph_file_path` is not provided. Default is None.
        directed : bool, optional
            Whether the graph is directed. Default is False.
        with_multihop : bool, optional
            Enables multihop interdependence in graph layers. Default is False.
        h : int, optional
            Number of hops for multihop interdependence. Default is 1.
        accumulative : bool, optional
            Accumulate interdependence over hops. Default is False.
        with_pagerank : bool, optional
            Include PageRank-based processing in layers. Default is False.
        c : float, optional
            Damping factor for PageRank. Default is 0.15.
        require_data : bool, optional
            Require data input for adjacency matrix processing. Default is False.
        require_parameters : bool, optional
            Require parameter input for adjacency matrix processing. Default is False.
        normalization : bool, optional
            Normalize adjacency matrix. Default is True.
        normalization_mode : str, optional
            Mode of normalization ('row' or 'column'). Default is 'column'.
        self_dependence : bool, optional
            Include self-loops in graph processing. Default is True.
        with_dual_lphm : bool, optional
            Enable dual LPHM parameter reconciliation. Default is False.
        with_lorr : bool, optional
            Enable LoRR parameter reconciliation. Default is False.
        r : int, optional
            Rank parameter for parameter reconciliation. Default is 3.
        with_residual : bool, optional
            Include residual connections in layers. Default is False.
        enable_bias : bool, optional
            Enable bias in parameter reconciliation. Default is False.
        with_batch_norm : bool, optional
            Include batch normalization in layers. Default is False.
        with_relu : bool, optional
            Include ReLU activation in layers. Default is True.
        with_softmax : bool, optional
            Include softmax activation in the output layer. Default is True.
        with_dropout : bool, optional
            Include dropout in layers. Default is False.
        p : float, optional
            Dropout probability. Default is 0.25.
        device : str, optional
            Device for computation ('cpu' or 'cuda'). Default is 'cpu'.
        *args : optional
            Additional positional arguments for superclass.
        **kwargs : optional
            Additional keyword arguments for superclass.

        Raises
        ------
        ValueError
            If `dims` contains fewer than two dimensions.

        """
        print('############# rpn-gcn model architecture ############')

        if len(dims) < 2:
            raise ValueError("At least two dim values is needed for defining the model...")

        layers = []
        for m, n in zip(dims, dims[1:]):
            print('layer: {},'.format(len(layers)), 'm: {}, n: {}'.format(m, n))
            layers.append(
                graph_interdependence_layer(
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

