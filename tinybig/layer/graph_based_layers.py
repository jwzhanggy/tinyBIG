# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

################################
# Graph based RPN Layer Module #
################################

"""
Graph Structural RPN based layers.

This module contains the graph structural rpn based layers, including
    chain_interdependence_layer

"""

from tinybig.module.base_layer import layer
from tinybig.head.graph_based_heads import graph_interdependence_head, graph_bilinear_interdependence_head
from tinybig.koala.topology import graph as graph_class


class graph_interdependence_layer(layer):
    """
    A layer that models interdependence in graph-structured data.

    This layer integrates multiple `graph_interdependence_head` instances to model relationships
    within a graph, supporting advanced features like multi-hop dependencies, PageRank-like structures,
    and flexible parameter reconciliation.

    Attributes
    ----------
    m : int
        The input dimensionality of the layer.
    n : int
        The output dimensionality of the layer.
    width : int
        The number of `graph_interdependence_head` instances in the layer.
    name : str
        The name of the layer.
    channel_num : int
        The number of channels in each `graph_interdependence_head`.
    graph : graph_class, optional
        The graph structure to be used. If not provided, `graph_file_path`, `nodes`, and `links` must be provided.
    graph_file_path : str, optional
        Path to the file from which the graph can be loaded.
    nodes : list, optional
        A list of node identifiers in the graph.
    links : list, optional
        A list of edges in the graph.
    directed : bool
        Whether the graph is directed.
    with_multihop : bool
        Whether to enable multi-hop dependencies.
    h : int
        Number of hops for multi-hop dependencies.
    accumulative : bool
        Whether to accumulate dependencies across hops.
    with_pagerank : bool
        Whether to include PageRank-like dependencies in the layer.
    c : float
        Damping factor for PageRank dependencies.
    require_data : bool
        Whether data is required for the graph interdependence function.
    require_parameters : bool
        Whether parameters are required for the graph interdependence function.
    normalization : bool
        Whether to normalize the adjacency matrix.
    normalization_mode : str
        The normalization mode for the adjacency matrix (`'row'` or `'column'`).
    self_dependence : bool
        Whether to include self-dependencies in the graph structure.
    with_dual_lphm : bool
        Whether to use dual LPHM reconciliation.
    with_lorr : bool
        Whether to use LORR reconciliation.
    r : int
        The rank for parameter reconciliation.
    with_residual : bool
        Whether to include residual connections in the layer.
    enable_bias : bool
        Whether to enable bias terms in parameter reconciliation.
    with_batch_norm : bool
        Whether to apply batch normalization to the layer's output.
    with_relu : bool
        Whether to apply ReLU activation to the layer's output.
    with_softmax : bool
        Whether to apply softmax activation to the layer's output.
    with_dropout : bool
        Whether to apply dropout to the layer's output.
    p : float
        The dropout probability.
    parameters_init_method : str
        The initialization method for parameters.
    device : str
        The device on which to run the layer (`'cpu'` or `'cuda'`).

    Methods
    -------
    __init__(...)
        Initializes the graph interdependence layer.
    """
    def __init__(
        self,
        m: int, n: int,
        width: int = 1,
        name: str = 'graph_interdependence_layer',
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
        with_dropout: bool = True, p: float = 0.5,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize the graph interdependence layer.

        This constructor creates a layer consisting of multiple graph interdependence heads
        to process and learn from graph-structured data.

        Parameters
        ----------
        m : int
            The input dimensionality of the layer.
        n : int
            The output dimensionality of the layer.
        width : int, optional
            The number of `graph_interdependence_head` instances in the layer. Default is 1.
        name : str, optional
            The name of the layer. Default is `'graph_interdependence_layer'`.
        channel_num : int, optional
            The number of channels in each head. Default is 1.
        graph : graph_class, optional
            A pre-loaded graph structure. If provided, this takes precedence over `graph_file_path`, `nodes`, and `links`.
        graph_file_path : str, optional
            Path to the file containing the graph structure.
        nodes : list, optional
            A list of node identifiers.
        links : list, optional
            A list of edges representing connections between nodes.
        directed : bool, optional
            Indicates whether the graph is directed. Default is `False`.
        with_multihop : bool, optional
            If `True`, enables multi-hop dependencies in the graph interdependence function. Default is `False`.
        h : int, optional
            The number of hops to consider for multi-hop dependencies. Default is 1.
        accumulative : bool, optional
            If `True`, accumulates dependencies across multiple hops. Default is `False`.
        with_pagerank : bool, optional
            If `True`, includes PageRank-like dependencies in the graph interdependence function. Default is `False`.
        c : float, optional
            Damping factor for PageRank dependencies. Default is 0.15.
        require_data : bool, optional
            If `True`, requires data for the graph interdependence function. Default is `False`.
        require_parameters : bool, optional
            If `True`, requires parameters for the graph interdependence function. Default is `False`.
        normalization : bool, optional
            If `True`, normalizes the adjacency matrix of the graph. Default is `True`.
        normalization_mode : str, optional
            Specifies the normalization mode (`'row'` or `'column'`). Default is `'column'`.
        self_dependence : bool, optional
            If `True`, includes self-loops (self-dependencies) in the graph structure. Default is `True`.
        with_dual_lphm : bool, optional
            If `True`, applies dual LPHM reconciliation for parameter fabrication. Default is `False`.
        with_lorr : bool, optional
            If `True`, applies LORR reconciliation for parameter fabrication. Default is `False`.
        r : int, optional
            The rank used for parameter reconciliation. Default is 3.
        with_residual : bool, optional
            If `True`, includes residual connections. Default is `False`.
        enable_bias : bool, optional
            If `True`, enables bias terms during parameter reconciliation. Default is `False`.
        with_batch_norm : bool, optional
            If `True`, applies batch normalization to the output. Default is `False`.
        with_relu : bool, optional
            If `True`, applies a ReLU activation to the output. Default is `True`.
        with_softmax : bool, optional
            If `True`, applies a softmax activation to the output. Default is `True`.
        with_dropout : bool, optional
            If `True`, applies dropout to the output. Default is `True`.
        p : float, optional
            The dropout probability. Default is 0.5.
        parameters_init_method : str, optional
            Specifies the initialization method for parameters. Default is `'xavier_normal'`.
        device : str, optional
            The device on which to run the layer (`'cpu'` or `'cuda'`). Default is `'cpu'`.

        Raises
        ------
        ValueError
            If a graph structure is not provided via `graph`, `graph_file_path`, `nodes`, or `links`.
        """
        print('* graph_interdependence_layer, width:', width)
        heads = [
            graph_interdependence_head(
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
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        print('--------------------------')
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class graph_bilinear_interdependence_layer(layer):
    """
    A layer that combines graph interdependence with bilinear interdependence.

    This layer integrates `graph_bilinear_interdependence_head` instances to simultaneously model
    graph-structured data relationships and bilinear dependencies.

    Attributes
    ----------
    m : int
        The input dimensionality of the layer.
    n : int
        The output dimensionality of the layer.
    width : int
        The number of `graph_bilinear_interdependence_head` instances in the layer.
    name : str
        The name of the layer.
    channel_num : int
        The number of channels in each `graph_bilinear_interdependence_head`.
    graph : graph_class, optional
        The graph structure to be used. If not provided, `graph_file_path`, `nodes`, and `links` must be provided.
    graph_file_path : str, optional
        Path to the file from which the graph can be loaded.
    nodes : list, optional
        A list of node identifiers in the graph.
    links : list, optional
        A list of edges in the graph.
    directed : bool
        Whether the graph is directed.
    with_multihop : bool
        Whether to enable multi-hop dependencies.
    h : int
        Number of hops for multi-hop dependencies.
    accumulative : bool
        Whether to accumulate dependencies across hops.
    with_pagerank : bool
        Whether to include PageRank-like dependencies in the layer.
    c : float
        Damping factor for PageRank dependencies.
    require_data : bool
        Whether data is required for the graph interdependence function.
    require_parameters : bool
        Whether parameters are required for the graph interdependence function.
    normalization : bool
        Whether to normalize the adjacency matrix.
    normalization_mode : str
        The normalization mode for the adjacency matrix (`'row'` or `'column'`).
    self_dependence : bool
        Whether to include self-dependencies in the graph structure.
    with_dual_lphm_interdependence : bool
        Whether to use dual LPHM bilinear interdependence.
    with_lorr_interdependence : bool
        Whether to use LORR bilinear interdependence.
    r_interdependence : int
        The rank for bilinear interdependence reconciliation.
    with_dual_lphm : bool
        Whether to use dual LPHM reconciliation.
    with_lorr : bool
        Whether to use LORR reconciliation.
    r : int
        The rank for parameter reconciliation.
    with_residual : bool
        Whether to include residual connections in the layer.
    enable_bias : bool
        Whether to enable bias terms in parameter reconciliation.
    with_batch_norm : bool
        Whether to apply batch normalization to the layer's output.
    with_relu : bool
        Whether to apply ReLU activation to the layer's output.
    with_softmax : bool
        Whether to apply softmax activation to the layer's output.
    with_dropout : bool
        Whether to apply dropout to the layer's output.
    p : float
        The dropout probability.
    parameters_init_method : str
        The initialization method for parameters.
    device : str
        The device on which to run the layer (`'cpu'` or `'cuda'`).

    Methods
    -------
    __init__(...)
        Initializes the graph bilinear interdependence layer.
    """
    def __init__(
        self,
        m: int, n: int,
        width: int = 1,
        name: str = 'graph_bilinear_interdependence_layer',
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
        with_dropout: bool = True, p: float = 0.5,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize the Graph Bilinear Interdependence Layer.

        This method creates a graph bilinear interdependence layer that combines graph-based
        dependencies with bilinear interdependence functions. The layer consists of multiple
        `graph_bilinear_interdependence_head` instances, which are configured based on the
        provided parameters.

        Parameters
        ----------
        m : int
            The input dimensionality of the layer.
        n : int
            The output dimensionality of the layer.
        width : int, optional
            The number of `graph_bilinear_interdependence_head` instances in the layer. Default is 1.
        name : str, optional
            The name of the layer. Default is `'graph_bilinear_interdependence_layer'`.
        channel_num : int, optional
            The number of channels in each head. Default is 1.
        graph : graph_class, optional
            A pre-loaded graph structure. If provided, this takes precedence over `graph_file_path`, `nodes`, and `links`.
        graph_file_path : str, optional
            Path to the file containing the graph structure.
        nodes : list, optional
            A list of node identifiers.
        links : list, optional
            A list of edges representing connections between nodes.
        directed : bool, optional
            Indicates whether the graph is directed. Default is `False`.
        with_multihop : bool, optional
            If `True`, enables multi-hop dependencies in the graph interdependence function. Default is `False`.
        h : int, optional
            The number of hops to consider for multi-hop dependencies. Default is 1.
        accumulative : bool, optional
            If `True`, accumulates dependencies across multiple hops. Default is `False`.
        with_pagerank : bool, optional
            If `True`, includes PageRank-like dependencies in the graph interdependence function. Default is `False`.
        c : float, optional
            Damping factor for PageRank dependencies. Default is 0.15.
        require_data : bool, optional
            If `True`, requires data for the graph interdependence function. Default is `False`.
        require_parameters : bool, optional
            If `True`, requires parameters for the graph interdependence function. Default is `False`.
        normalization : bool, optional
            If `True`, normalizes the adjacency matrix of the graph. Default is `False`.
        normalization_mode : str, optional
            Specifies the normalization mode (`'row'` or `'column'`). Default is `'column'`.
        self_dependence : bool, optional
            If `True`, includes self-loops (self-dependencies) in the graph structure. Default is `True`.
        with_dual_lphm_interdependence : bool, optional
            If `True`, uses dual LPHM bilinear interdependence. Default is `False`.
        with_lorr_interdependence : bool, optional
            If `True`, uses LORR bilinear interdependence. Default is `False`.
        r_interdependence : int, optional
            The rank for bilinear interdependence parameterization. Default is 3.
        with_dual_lphm : bool, optional
            If `True`, applies dual LPHM reconciliation for parameter fabrication. Default is `False`.
        with_lorr : bool, optional
            If `True`, applies LORR reconciliation for parameter fabrication. Default is `False`.
        r : int, optional
            The rank used for parameter reconciliation. Default is 3.
        with_residual : bool, optional
            If `True`, includes residual connections. Default is `False`.
        enable_bias : bool, optional
            If `True`, enables bias terms during parameter reconciliation. Default is `False`.
        with_batch_norm : bool, optional
            If `True`, applies batch normalization to the output. Default is `False`.
        with_relu : bool, optional
            If `True`, applies a ReLU activation to the output. Default is `True`.
        with_softmax : bool, optional
            If `True`, applies a softmax activation to the output. Default is `True`.
        with_dropout : bool, optional
            If `True`, applies dropout to the output. Default is `True`.
        p : float, optional
            The dropout probability. Default is 0.5.
        parameters_init_method : str, optional
            Specifies the initialization method for parameters. Default is `'xavier_normal'`.
        device : str, optional
            The device on which to run the layer (`'cpu'` or `'cuda'`). Default is `'cpu'`.

        Raises
        ------
        ValueError
            If a graph structure is not provided via `graph`, `graph_file_path`, `nodes`, or `links`.

        Notes
        -----
        - This layer supports both graph-based interdependence and bilinear interdependence,
          combining the two through a fusion mechanism.
        - If both `graph` and `graph_file_path` are provided, `graph` takes precedence.
        """
        print('* graph_bilinear_interdependence_layer, width:', width)
        heads = [
            graph_bilinear_interdependence_head(
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
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        print('--------------------------')
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)
