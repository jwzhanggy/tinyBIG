# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# RPN Multi-Head Layer Module #
###############################

"""
Basic RPN based layers.

This module contains the basic rpn based layers, including
    perceptron_layer,
    svm_layer,
    kan_layer,
    pgm_layer,
    naive_bayes_layer
"""

from tinybig.module.base_layer import layer
from tinybig.head.basic_heads import perceptron_head, svm_head, kan_head, naive_bayes_head, pgm_head


class perceptron_layer(layer):
    """
    A layer consisting of perceptron heads.

    This layer uses perceptron heads with optional data expansion, parameter reconciliation, and output processing.

    Parameters
    ----------
    m : int
        The input dimension of the perceptron heads.
    n : int
        The output dimension of the perceptron heads.
    name : str, default='perceptron_layer'
        The name of the layer.
    channel_num : int, default=1
        The number of channels in the layer.
    width : int, default=1
        The number of perceptron heads in the layer.
    with_bspline : bool, default=False
        Whether to use B-spline expansion.
    with_taylor : bool, default=False
        Whether to use Taylor expansion.
    d : int, default=2
        The degree of expansion when using Taylor or B-spline expansion.
    with_hybrid_expansion : bool, default=False
        Whether to use hybrid expansion.
    with_dual_lphm : bool, default=False
        Whether to use dual LPHM reconciliation.
    with_lorr : bool, default=False
        Whether to use LORR reconciliation.
    r : int, default=3
        The rank for parameter reconciliation.
    enable_bias : bool, default=True
        Whether to enable bias in parameter reconciliation.
    with_residual : bool, default=False
        Whether to include a residual connection.
    with_batch_norm : bool, default=False
        Whether to apply batch normalization.
    with_relu : bool, default=True
        Whether to apply ReLU activation.
    with_dropout : bool, default=True
        Whether to apply dropout.
    p : float, default=0.5
        Dropout probability.
    with_softmax : bool, default=True
        Whether to apply softmax activation.
    parameters_init_method : str, default='xavier_normal'
        The method for parameter initialization.
    device : str, default='cpu'
        The device to run the layer on ('cpu' or 'cuda').

    Returns
    -------
    perceptron_layer
        An initialized perceptron layer with the specified configuration.
    """

    def __init__(
        self,
        m: int, n: int,
        name: str = 'perceptron_layer',
        channel_num: int = 1,
        width: int = 1,
        # data expansion function
        with_bspline: bool = False,
        with_taylor: bool = False, d: int = 2,
        with_hybrid_expansion: bool = False,
        # parameter reconciliation function parameters
        with_dual_lphm: bool = False,
        with_lorr: bool = False, r: int = 3,
        enable_bias: bool = True,
        # remainder function parameters
        with_residual: bool = False,
        # output processing function parameters
        with_batch_norm: bool = False,
        with_relu: bool = True,
        with_dropout: bool = True, p: float = 0.5,
        with_softmax: bool = True,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize a perceptron layer.

        Parameters
        ----------
        m : int
            The input dimension of the perceptron heads.
        n : int
            The output dimension of the perceptron heads.
        name : str, default='perceptron_layer'
            The name of the layer.
        channel_num : int, default=1
            The number of channels in the layer.
        width : int, default=1
            The number of perceptron heads in the layer.
        with_bspline : bool, default=False
            Whether to use B-spline expansion.
        with_taylor : bool, default=False
            Whether to use Taylor expansion.
        d : int, default=2
            The degree of expansion when using Taylor or B-spline expansion.
        with_hybrid_expansion : bool, default=False
            Whether to use hybrid expansion.
        with_dual_lphm : bool, default=False
            Whether to use dual LPHM reconciliation.
        with_lorr : bool, default=False
            Whether to use LORR reconciliation.
        r : int, default=3
            The rank for parameter reconciliation.
        enable_bias : bool, default=True
            Whether to enable bias in parameter reconciliation.
        with_residual : bool, default=False
            Whether to include a residual connection.
        with_batch_norm : bool, default=False
            Whether to apply batch normalization.
        with_relu : bool, default=True
            Whether to apply ReLU activation.
        with_dropout : bool, default=True
            Whether to apply dropout.
        p : float, default=0.5
            Dropout probability.
        with_softmax : bool, default=True
            Whether to apply softmax activation.
        parameters_init_method : str, default='xavier_normal'
            The method for parameter initialization.
        device : str, default='cpu'
            The device to run the layer on ('cpu' or 'cuda').

        Returns
        -------
        None
        """
        print('* perceptron_layer, width:', width)
        heads = [
            perceptron_head(
                m=m, n=n,
                channel_num=channel_num,
                # --------------------
                with_bspline=with_bspline,
                with_taylor=with_taylor, d=d,
                with_hybrid_expansion=with_hybrid_expansion,
                # --------------------
                with_dual_lphm=with_dual_lphm,
                with_lorr=with_lorr, r=r,
                enable_bias=enable_bias,
                # --------------------
                with_residual=with_residual,
                # --------------------
                with_batch_norm=with_batch_norm,
                with_relu=with_relu,
                with_dropout=with_dropout, p=p,
                with_softmax=with_softmax,
                # --------------------
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        print('--------------------------')
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class svm_layer(layer):
    """
    A layer consisting of SVM heads.

    This layer uses SVM heads with specified kernels and optional parameter reconciliation.

    Parameters
    ----------
    m : int
        The input dimension of the SVM heads.
    n : int
        The output dimension of the SVM heads.
    name : str, default='svm_layer'
        The name of the layer.
    kernel : str, default='linear'
        The kernel function to use ('linear', 'gaussian_rbf', or 'inverse_quadratic_rbf').
    base_range : tuple, default=(-1, 1)
        The base range for kernel functions.
    num_interval : int, default=10
        The number of intervals for kernel functions.
    epsilon : float, default=1.0
        The epsilon parameter for kernel functions.
    enable_bias : bool, default=False
        Whether to enable bias in the heads.
    with_lorr : bool, default=False
        Whether to use LORR reconciliation.
    r : int, default=3
        The rank for parameter reconciliation.
    with_residual : bool, default=False
        Whether to include a residual connection.
    channel_num : int, default=1
        The number of channels in the layer.
    width : int, default=1
        The number of SVM heads in the layer.
    parameters_init_method : str, default='xavier_normal'
        The method for parameter initialization.
    device : str, default='cpu'
        The device to run the layer on ('cpu' or 'cuda').

    Returns
    -------
    svm_layer
        An initialized SVM layer with the specified configuration.
    """
    def __init__(
        self, m: int, n: int,
        name: str = 'svm_layer',
        kernel: str = 'linear',
        base_range: tuple = (-1, 1),
        num_interval: int = 10,
        epsilon: float = 1.0,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize an SVM layer.

        Parameters
        ----------
        m : int
            The input dimension of the SVM heads.
        n : int
            The output dimension of the SVM heads.
        name : str, default='svm_layer'
            The name of the layer.
        kernel : str, default='linear'
            The kernel function to use ('linear', 'gaussian_rbf', or 'inverse_quadratic_rbf').
        base_range : tuple, default=(-1, 1)
            The base range for kernel functions.
        num_interval : int, default=10
            The number of intervals for kernel functions.
        epsilon : float, default=1.0
            The epsilon parameter for kernel functions.
        enable_bias : bool, default=False
            Whether to enable bias in the heads.
        with_lorr : bool, default=False
            Whether to use LORR reconciliation.
        r : int, default=3
            The rank for parameter reconciliation.
        with_residual : bool, default=False
            Whether to include a residual connection.
        channel_num : int, default=1
            The number of channels in the layer.
        width : int, default=1
            The number of SVM heads in the layer.
        parameters_init_method : str, default='xavier_normal'
            The method for parameter initialization.
        device : str, default='cpu'
            The device to run the layer on ('cpu' or 'cuda').

        Returns
        -------
        None
        """
        heads = [
            svm_head(
                m=m, n=n,
                kernel=kernel,
                base_range=base_range,
                num_interval=num_interval,
                epsilon=epsilon,
                enable_bias=enable_bias,
                with_lorr=with_lorr, r=r,
                with_residual=with_residual,
                channel_num=channel_num,
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class kan_layer(layer):
    """
    A layer consisting of KAN heads.

    This layer uses KAN heads with B-spline expansion and optional parameter reconciliation.

    Parameters
    ----------
    m : int
        The input dimension of the KAN heads.
    n : int
        The output dimension of the KAN heads.
    grid_range : tuple, default=(-1, 1)
        The grid range for the B-spline expansion.
    t : int, default=5
        The number of grid points for the B-spline expansion.
    d : int, default=3
        The degree of the B-spline expansion.
    name : str, default='kan_layer'
        The name of the layer.
    enable_bias : bool, default=False
        Whether to enable bias in parameter reconciliation.
    with_lorr : bool, default=False
        Whether to use LORR reconciliation.
    r : int, default=3
        The rank for parameter reconciliation.
    channel_num : int, default=1
        The number of channels in the layer.
    width : int, default=1
        The number of KAN heads in the layer.
    parameters_init_method : str, default='xavier_normal'
        The method for parameter initialization.
    device : str, default='cpu'
        The device to run the layer on ('cpu' or 'cuda').

    Returns
    -------
    kan_layer
        An initialized KAN layer with the specified configuration.
    """
    def __init__(
        self, m: int, n: int,
        grid_range=(-1, 1), t: int = 5, d: int = 3,
        name: str = 'perceptron_head',
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False, r: int = 3,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize a KAN layer.

        Parameters
        ----------
        m : int
            The input dimension of the KAN heads.
        n : int
            The output dimension of the KAN heads.
        grid_range : tuple, default=(-1, 1)
            The grid range for the B-spline expansion.
        t : int, default=5
            The number of grid points for the B-spline expansion.
        d : int, default=3
            The degree of the B-spline expansion.
        name : str, default='kan_layer'
            The name of the layer.
        enable_bias : bool, default=False
            Whether to enable bias in parameter reconciliation.
        with_lorr : bool, default=False
            Whether to use LORR reconciliation.
        r : int, default=3
            The rank for parameter reconciliation.
        channel_num : int, default=1
            The number of channels in the layer.
        width : int, default=1
            The number of KAN heads in the layer.
        parameters_init_method : str, default='xavier_normal'
            The method for parameter initialization.
        device : str, default='cpu'
            The device to run the layer on ('cpu' or 'cuda').

        Returns
        -------
        None
        """
        heads = [
            kan_head(
                m=m, n=n,
                grid_range=grid_range,
                t=t, d=d,
                enable_bias=enable_bias,
                with_lorr=with_lorr, r=r,
                channel_num=channel_num,
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class naive_bayes_layer(layer):
    """
    A layer consisting of Naive Bayes heads.

    This layer uses Naive Bayes heads for probabilistic modeling.

    Parameters
    ----------
    m : int
        The input dimension of the Naive Bayes heads.
    n : int
        The output dimension of the Naive Bayes heads.
    name : str, default='naive_bayes_layer'
        The name of the layer.
    distribution : str, default='normal'
        The distribution to use ('normal', 'exponential', 'gamma', etc.).
    enable_bias : bool, default=False
        Whether to enable bias in parameter reconciliation.
    with_lorr : bool, default=False
        Whether to use LORR reconciliation.
    r : int, default=3
        The rank for parameter reconciliation.
    with_residual : bool, default=False
        Whether to include a residual connection.
    channel_num : int, default=1
        The number of channels in the layer.
    width : int, default=1
        The number of Naive Bayes heads in the layer.
    parameters_init_method : str, default='xavier_normal'
        The method for parameter initialization.
    device : str, default='cpu'
        The device to run the layer on ('cpu' or 'cuda').

    Returns
    -------
    naive_bayes_layer
        An initialized Naive Bayes layer with the specified configuration.
    """
    def __init__(
        self,
        m: int, n: int,
        name: str = 'perceptron_layer',
        distribution: str = 'normal',
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize a Naive Bayes layer.

        Parameters
        ----------
        m : int
            The input dimension of the Naive Bayes heads.
        n : int
            The output dimension of the Naive Bayes heads.
        name : str, default='naive_bayes_layer'
            The name of the layer.
        distribution : str, default='normal'
            The distribution to use ('normal', 'exponential', 'gamma', etc.).
        enable_bias : bool, default=False
            Whether to enable bias in parameter reconciliation.
        with_lorr : bool, default=False
            Whether to use LORR reconciliation.
        r : int, default=3
            The rank for parameter reconciliation.
        with_residual : bool, default=False
            Whether to include a residual connection.
        channel_num : int, default=1
            The number of channels in the layer.
        width : int, default=1
            The number of Naive Bayes heads in the layer.
        parameters_init_method : str, default='xavier_normal'
            The method for parameter initialization.
        device : str, default='cpu'
            The device to run the layer on ('cpu' or 'cuda').

        Returns
        -------
        None
        """
        heads = [
            naive_bayes_head(
                m=m, n=n,
                enable_bias=enable_bias,
                distribution=distribution,
                with_lorr=with_lorr, r=r,
                with_residual=with_residual,
                channel_num=channel_num,
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)


class pgm_layer(layer):
    """
    A layer consisting of Probabilistic Graphical Model (PGM) heads.

    This layer uses PGM heads for structured probabilistic modeling.

    Parameters
    ----------
    m : int
        The input dimension of the PGM heads.
    n : int
        The output dimension of the PGM heads.
    name : str, default='pgm_layer'
        The name of the layer.
    distribution : str, default='normal'
        The distribution to use ('normal', 'exponential', 'gamma', etc.).
    d : int, default=2
        The degree of the combinatorial expansion.
    with_replacement : bool, default=False
        Whether to allow replacement in the combinatorial expansion.
    enable_bias : bool, default=False
        Whether to enable bias in parameter reconciliation.
    with_lorr : bool, default=False
        Whether to use LORR reconciliation.
    r : int, default=3
        The rank for parameter reconciliation.
    with_residual : bool, default=False
        Whether to include a residual connection.
    channel_num : int, default=1
        The number of channels in the layer.
    width : int, default=1
        The number of PGM heads in the layer.
    parameters_init_method : str, default='xavier_normal'
        The method for parameter initialization.
    device : str, default='cpu'
        The device to run the layer on ('cpu' or 'cuda').

    Returns
    -------
    pgm_layer
        An initialized PGM layer with the specified configuration.
    """
    def __init__(
        self,
        m: int, n: int,
        name: str = 'perceptron_layer',
        distribution: str = 'normal',
        d: int = 2, with_replacement: bool = False,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        width: int = 1,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initialize a Probabilistic Graphical Model (PGM) layer.

        Parameters
        ----------
        m : int
            The input dimension of the PGM heads.
        n : int
            The output dimension of the PGM heads.
        name : str, default='pgm_layer'
            The name of the layer.
        distribution : str, default='normal'
            The distribution to use ('normal', 'exponential', 'gamma', etc.).
        d : int, default=2
            The degree of the combinatorial expansion.
        with_replacement : bool, default=False
            Whether to allow replacement in the combinatorial expansion.
        enable_bias : bool, default=False
            Whether to enable bias in parameter reconciliation.
        with_lorr : bool, default=False
            Whether to use LORR reconciliation.
        r : int, default=3
            The rank for parameter reconciliation.
        with_residual : bool, default=False
            Whether to include a residual connection.
        channel_num : int, default=1
            The number of channels in the layer.
        width : int, default=1
            The number of PGM heads in the layer.
        parameters_init_method : str, default='xavier_normal'
            The method for parameter initialization.
        device : str, default='cpu'
            The device to run the layer on ('cpu' or 'cuda').

        Returns
        -------
        None
        """
        heads = [
            pgm_head(
                m=m, n=n,
                enable_bias=enable_bias,
                distribution=distribution,
                d=d, with_replacement=with_replacement,
                with_lorr=with_lorr, r=r,
                with_residual=with_residual,
                channel_num=channel_num,
                parameters_init_method=parameters_init_method,
                device=device, *args, **kwargs
            )
        ] * width
        super().__init__(name=name, m=m, n=n, heads=heads, device=device, *args, **kwargs)
