# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

########################
# RPN based GNN Models #
########################

"""
RPN based GNN models

This module contains the implementation of the RPN based GNN models, including
    gat
"""

from tinybig.model.rpn import rpn
from tinybig.layer import perceptron_layer
from tinybig.layer.graph_based_layers import graph_bilinear_interdependence_layer
from tinybig.koala.topology import graph as graph_class


class gat(rpn):
    """
    Graph Attention Network (GAT) implemented using the RPN framework.

    This GAT model leverages graph bilinear interdependence layers to capture graph-based relationships,
    enabling efficient learning on graph-structured data. The architecture is highly customizable,
    supporting multihop interdependence, PageRank-based processing, and advanced parameter reconciliation.

    Attributes
    ----------
    dims : list[int] | tuple[int]
        List of dimensions for the layers in the GAT. Must contain at least two positive integers.
    channel_num : int
        Number of channels for the interdependence layers. Default is 1.
    width : int
        Number of parallel heads in each layer. Default is 1.
    name : str
        Name of the GAT model. Default is 'rpn_gcn'.
    graph : graph_class, optional
        A pre-constructed graph structure. Default is None.
    graph_file_path : str, optional
        Path to a file containing graph data. Default is None.
    nodes : list, optional
        List of nodes in the graph. Required if `graph` or `graph_file_path` is not provided. Default is None.
    links : list, optional
        List of edges in the graph. Required if `graph` or `graph_file_path` is not provided. Default is None.
    directed : bool
        Whether the graph is directed. Default is False.
    with_multihop : bool
        Enables multihop interdependence in the graph layers. Default is False.
    h : int
        Number of hops for multihop interdependence. Default is 1.
    accumulative : bool
        If True, accumulates interdependence over hops. Default is False.
    with_pagerank : bool
        If True, includes PageRank-based processing in the layers. Default is False.
    c : float
        Damping factor for PageRank. Default is 0.15.
    require_data : bool
        If True, requires data input for the adjacency matrix processing. Default is False.
    require_parameters : bool
        If True, requires parameter input for the adjacency matrix processing. Default is False.
    normalization : bool
        Whether to normalize the adjacency matrix. Default is True.
    normalization_mode : str
        Mode of normalization ('row' or 'column'). Default is 'column'.
    self_dependence : bool
        Whether to include self-loops in the graph processing. Default is True.
    with_dual_lphm_interdependence : bool
        If True, uses dual LPHM (Low-Parameter High-Model) interdependence. Default is False.
    with_lorr_interdependence : bool
        If True, uses Low-Rank Reconciliation (LoRR) interdependence. Default is False.
    r_interdependence : int
        Rank parameter for bilinear interdependence functions. Default is 3.
    with_dual_lphm : bool
        If True, enables dual LPHM parameter reconciliation. Default is False.
    with_lorr : bool
        If True, enables LoRR parameter reconciliation. Default is False.
    r : int
        Rank parameter for parameter reconciliation. Default is 3.
    with_residual : bool
        If True, includes residual connections in the layers. Default is False.
    enable_bias : bool
        Whether to enable bias in parameter reconciliation. Default is False.
    with_batch_norm : bool
        Whether to include batch normalization in the layers. Default is False.
    with_relu : bool
        Whether to include ReLU activation in the layers. Default is True.
    with_softmax : bool
        Whether to include softmax activation in the output layer. Default is True.
    with_dropout : bool
        Whether to include dropout in the layers. Default is False.
    p : float
        Dropout probability. Default is 0.25.
    device : str
        Device for computation ('cpu' or 'cuda'). Default is 'cpu'.

    Methods
    -------
    __init__(...)
        Initializes the GAT model, constructing graph bilinear interdependence layers based on the specified parameters.
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
        with_dropout: bool = False, p: float = 0.25,
        # other parameters
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize the GAT (Graph Attention Network) model within the RPN (Recurrent Patch Network) framework.

        Parameters
        ----------
        dims : list[int] | tuple[int]
            List of layer dimensions for the GAT. Must have at least two positive integers.
        channel_num : int, optional
            Number of channels for interdependence layers. Default is 1.
        width : int, optional
            Number of parallel heads in each layer. Default is 1.
        name : str, optional
            Name of the GAT model. Default is 'rpn_gcn'.
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
        with_dual_lphm_interdependence : bool, optional
            Use dual LPHM interdependence in layers. Default is False.
        with_lorr_interdependence : bool, optional
            Use LoRR interdependence in layers. Default is False.
        r_interdependence : int, optional
            Rank parameter for bilinear interdependence. Default is 3.
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
            If `dims` is empty or has less than two dimensions.
            If any value in `dims` is not a positive integer.

        """
        print('############# rpn-gat model architecture ############')

        if dims is None or len(dims) < 2:
            raise ValueError('dims must not be empty and need to have at least two dimensions...')
        assert all(isinstance(d, int) and d > 0 for d in dims)

        layers = []
        for m, n in zip(dims[0:], dims[1:]):
            print('layer: {},'.format(len(layers)), 'm: {}, n: {}'.format(m, n))
            layers.append(
                graph_bilinear_interdependence_layer(
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
                    with_dual_lphm_interdependence=with_dual_lphm_interdependence,
                    with_lorr_interdependence=with_lorr_interdependence, r_interdependence=r_interdependence,
                    # ---------------
                    with_dual_lphm=with_dual_lphm,
                    with_lorr=with_lorr, r=r,
                    enable_bias=enable_bias,
                    # ---------------
                    with_residual=with_residual,
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

