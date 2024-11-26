# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# RPN Multi-Channel Head Modules #
##################################

"""
Basic RPN based heads.

This module contains the basic rpn based heads, including
    perceptron_head,
    svm_head,
    kan_head,
    pgm_head,
    naive_bayes_head,
"""

import torch

from tinybig.module.base_head import head

from tinybig.expansion import (
    identity_expansion,
    taylor_expansion,
    bspline_expansion,
    gaussian_rbf_expansion,
    inverse_quadratic_rbf_expansion,
    naive_gamma_expansion,
    naive_cauchy_expansion,
    naive_normal_expansion,
    naive_laplace_expansion,
    naive_exponential_expansion,
    naive_chi2_expansion,
    combinatorial_normal_expansion,
)

from tinybig.reconciliation import identity_reconciliation, lorr_reconciliation, dual_lphm_reconciliation
from tinybig.remainder import zero_remainder, linear_remainder


class perceptron_head(head):
    """
    A perceptron-based head for implementing multi-channel modules.

    The perceptron head is designed to work with various data transformations, parameter reconciliation,
    and remainder functions. It also supports optional features such as dropout, batch normalization, and activation functions.

    Attributes
    ----------
    m : int
        Input dimension of the head.
    n : int
        Output dimension of the head.
    channel_num : int
        Number of channels for multi-channel processing.
    parameters_init_method : str
        Initialization method for parameters.
    device : str
        Device to host the head (e.g., 'cpu' or 'cuda').

    Methods
    -------
    __init__(...)
        Initializes the perceptron head with specified configurations.
    """
    def __init__(
        self, m: int, n: int,
        name: str = 'perceptron_head',
        channel_num: int = 1,
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
        with_relu: bool = False,
        with_dropout: bool = True, p: float = 0.5,
        with_softmax: bool = False,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the perceptron head.

        Parameters
        ----------
        m : int
            Input dimension.
        n : int
            Output dimension.
        name : str, optional
            Name of the perceptron head, default is 'perceptron_head'.
        channel_num : int, optional
            Number of channels for processing, default is 1.
        with_bspline : bool, optional
            Whether to use B-spline expansion, default is False.
        with_taylor : bool, optional
            Whether to use Taylor expansion, default is False.
        d : int, optional
            Degree for the expansion functions, default is 2.
        with_hybrid_expansion : bool, optional
            Whether to use hybrid expansion, default is False.
        with_dual_lphm : bool, optional
            Whether to use dual LPHM reconciliation, default is False.
        with_lorr : bool, optional
            Whether to use LORR reconciliation, default is False.
        r : int, optional
            Parameter for reconciliation functions, default is 3.
        enable_bias : bool, optional
            Whether to enable bias in reconciliation functions, default is True.
        with_residual : bool, optional
            Whether to include a residual connection, default is False.
        with_batch_norm : bool, optional
            Whether to include batch normalization, default is False.
        with_relu : bool, optional
            Whether to use ReLU activation, default is False.
        with_dropout : bool, optional
            Whether to include dropout, default is True.
        p : float, optional
            Dropout probability, default is 0.5.
        with_softmax : bool, optional
            Whether to use softmax activation, default is False.
        parameters_init_method : str, optional
            Initialization method for parameters, default is 'xavier_normal'.
        device : str, optional
            Device to host the head, default is 'cpu'.

        Returns
        -------
        None
        """
        if with_taylor:
            data_transformation = taylor_expansion(
                d=d,
                device=device,
            )
        elif with_bspline:
            data_transformation = bspline_expansion(
                d=d,
                device=device,
            )
        else:
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
            m=m, n=n, name=name,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            channel_num=channel_num,
            parameters_init_method=parameters_init_method,
            device=device, *args, **kwargs
        )


class svm_head(head):
    """
    A support vector machine (SVM)-based head for implementing multi-channel modules.

    This head supports linear, Gaussian RBF, and inverse quadratic RBF kernels, along with
    optional parameter reconciliation, residual connections, and output processing functions.

    Attributes
    ----------
    m : int
        Input dimension of the head.
    n : int
        Output dimension of the head.
    kernel : str
        Type of kernel function ('linear', 'gaussian_rbf', 'inverse_quadratic_rbf').
    channel_num : int
        Number of channels for multi-channel processing.
    parameters_init_method : str
        Initialization method for parameters.
    device : str
        Device to host the head (e.g., 'cpu' or 'cuda').

    Methods
    -------
    __init__(...)
        Initializes the SVM head with specified configurations.
    """
    def __init__(
        self, m: int, n: int,
        name: str = 'svm_head',
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
        with_batch_norm: bool = False,
        with_softmax: bool = False,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the SVM head.

        Parameters
        ----------
        m : int
            Input dimension.
        n : int
            Output dimension.
        name : str, optional
            Name of the SVM head, default is 'svm_head'.
        kernel : str, optional
            Type of kernel function ('linear', 'gaussian_rbf', 'inverse_quadratic_rbf'), default is 'linear'.
        base_range : tuple, optional
            Range for Gaussian RBF kernels, default is (-1, 1).
        num_interval : int, optional
            Number of intervals for the kernel, default is 10.
        epsilon : float, optional
            Epsilon value for kernels, default is 1.0.
        enable_bias : bool, optional
            Whether to enable bias in reconciliation functions, default is False.
        with_lorr : bool, optional
            Whether to use LORR reconciliation, default is False.
        r : int, optional
            Parameter for reconciliation functions, default is 3.
        with_residual : bool, optional
            Whether to include a residual connection, default is False.
        channel_num : int, optional
            Number of channels, default is 1.
        with_batch_norm : bool, optional
            Whether to include batch normalization, default is False.
        with_softmax : bool, optional
            Whether to use softmax activation, default is False.
        parameters_init_method : str, optional
            Initialization method for parameters, default is 'xavier_normal'.
        device : str, optional
            Device to host the head, default is 'cpu'.

        Returns
        -------
        None
        """
        if kernel == 'linear':
            data_transformation = identity_expansion(
                device=device,
            )
        elif kernel == 'gaussian_rbf':
            data_transformation = gaussian_rbf_expansion(
                base_range=base_range,
                num_interval=num_interval,
                epsilon=epsilon,
                device=device,
            )
        elif kernel == 'inverse_quadratic_rbf':
            data_transformation = inverse_quadratic_rbf_expansion(
                base_range=base_range,
                num_interval=num_interval,
                epsilon=epsilon,
                device=device,
            )
        else:
            raise ValueError('kernel must be linear or gaussian_rbf or inverse_quadratic_rbf...')

        if with_lorr:
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

        if with_residual:
            remainder = linear_remainder(
                device=device
            )
        else:
            remainder = zero_remainder(
                device=device,
            )

        output_process_functions = []
        if with_batch_norm:
            output_process_functions.append(torch.nn.BatchNorm1d(num_features=n, device=device))
        if with_softmax:
            output_process_functions.append(torch.nn.Softmax(dim=-1))

        super().__init__(
            m=m, n=n, name=name,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            channel_num=channel_num,
            parameters_init_method=parameters_init_method,
            device=device, *args, **kwargs
        )


class kan_head(head):
    """
    A knowledge-aware network (KAN)-based head using B-spline expansion.

    Supports B-spline-based data transformation, parameter reconciliation, and flexible output processing.

    Attributes
    ----------
    m : int
        Input dimension of the head.
    n : int
        Output dimension of the head.
    grid_range : tuple
        Range for B-spline grid.
    t : int
        Number of knots for B-spline.
    d : int
        Degree of the B-spline.
    channel_num : int
        Number of channels for multi-channel processing.
    parameters_init_method : str
        Initialization method for parameters.
    device : str
        Device to host the head (e.g., 'cpu' or 'cuda').

    Methods
    -------
    __init__(...)
        Initializes the KAN head with specified configurations.
    """
    def __init__(
        self, m: int, n: int,
        grid_range=(-1, 1), t: int = 5, d: int = 3,
        name: str = 'kan_head',
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False, r: int = 3,
        channel_num: int = 1,
        with_batch_norm: bool = False,
        with_softmax: bool = False,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the KAN head.

        Parameters
        ----------
        m : int
            Input dimension.
        n : int
            Output dimension.
        grid_range : tuple, optional
            Range for B-spline grid, default is (-1, 1).
        t : int, optional
            Number of knots for B-spline, default is 5.
        d : int, optional
            Degree of the B-spline, default is 3.
        name : str, optional
            Name of the KAN head, default is 'kan_head'.
        enable_bias : bool, optional
            Whether to enable bias in reconciliation functions, default is False.
        with_lorr : bool, optional
            Whether to use LORR reconciliation, default is False.
        r : int, optional
            Parameter for reconciliation functions, default is 3.
        channel_num : int, optional
            Number of channels, default is 1.
        with_batch_norm : bool, optional
            Whether to include batch normalization, default is False.
        with_softmax : bool, optional
            Whether to use softmax activation, default is False.
        parameters_init_method : str, optional
            Initialization method for parameters, default is 'xavier_normal'.
        device : str, optional
            Device to host the head, default is 'cpu'.

        Returns
        -------
        None
        """
        data_transformation = bspline_expansion(
            grid_range=grid_range,
            t=t, d=d,
            device=device,
        )

        if with_lorr:
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

        remainder = linear_remainder(
            require_remainder_parameters=True,
            activation_functions=[torch.nn.SiLU()],
            device=device,
        )

        output_process_functions = []
        if with_batch_norm:
            output_process_functions.append(torch.nn.BatchNorm1d(num_features=n, device=device))
        if with_softmax:
            output_process_functions.append(torch.nn.Softmax(dim=-1))

        super().__init__(
            m=m, n=n, name=name,
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            output_process_functions=output_process_functions,
            channel_num=channel_num,
            parameters_init_method=parameters_init_method,
            device=device, *args, **kwargs
        )


class naive_bayes_head(head):
    """
    A Naive Bayes-based head for implementing multi-channel modules.

    This head supports various probability distributions, including normal, exponential, cauchy, gamma, chi-squared,
    and Laplace. It also allows for optional parameter reconciliation and residual connections.

    Attributes
    ----------
    m : int
        Input dimension of the head.
    n : int
        Output dimension of the head.
    distribution : str
        Type of probability distribution ('normal', 'exponential', 'cauchy', 'gamma', 'chi2', 'laplace').
    channel_num : int
        Number of channels for multi-channel processing.
    parameters_init_method : str
        Initialization method for parameters.
    device : str
        Device to host the head (e.g., 'cpu' or 'cuda').

    Methods
    -------
    __init__(...)
        Initializes the Naive Bayes head with specified configurations.
    """
    def __init__(
        self, m: int, n: int,
        name: str = 'perceptron_head',
        distribution: str = 'normal',
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the Naive Bayes head.

        Parameters
        ----------
        m : int
            Input dimension.
        n : int
            Output dimension.
        name : str, optional
            Name of the Naive Bayes head, default is 'naive_bayes_head'.
        distribution : str, optional
            Probability distribution ('normal', 'exponential', 'cauchy', 'gamma', 'chi2', 'laplace'), default is 'normal'.
        enable_bias : bool, optional
            Whether to enable bias in reconciliation functions, default is False.
        with_lorr : bool, optional
            Whether to use LORR reconciliation, default is False.
        r : int, optional
            Parameter for reconciliation functions, default is 3.
        with_residual : bool, optional
            Whether to include a residual connection, default is False.
        channel_num : int, optional
            Number of channels for multi-channel processing, default is 1.
        parameters_init_method : str, optional
            Initialization method for parameters, default is 'xavier_normal'.
        device : str, optional
            Device to host the head, default is 'cpu'.

        Returns
        -------
        None
        """
        if distribution == 'normal':
            data_transformation = naive_normal_expansion(
                device=device,
            )
        elif distribution == 'exponential':
            data_transformation = naive_exponential_expansion(
                device=device,
            )
        elif distribution == 'cauchy':
            data_transformation = naive_cauchy_expansion(
                device=device,
            )
        elif distribution == 'gamma':
            data_transformation = naive_gamma_expansion(
                device=device,
            )
        elif distribution == 'chi2':
            data_transformation = naive_chi2_expansion(
                device=device,
            )
        elif distribution == 'laplace':
            data_transformation = naive_laplace_expansion(
                device=device,
            )
        else:
            raise ValueError('tinybig only supports normal, exponential, cauchy, gamma, laplace or chi2 distributions...')

        if with_lorr:
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
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            channel_num=channel_num,
            parameters_init_method=parameters_init_method,
            device=device, *args, **kwargs
        )


class pgm_head(head):
    """
    A probabilistic graphical model (PGM)-based head for implementing multi-channel modules.

    This head supports combinatorial normal expansion and optional parameter reconciliation
    and residual connections. It is tailored for use with probabilistic graphical models.

    Attributes
    ----------
    m : int
        Input dimension of the head.
    n : int
        Output dimension of the head.
    d : int
        Degree for combinatorial expansion.
    with_replacement : bool
        Whether combinations are generated with replacement.
    channel_num : int
        Number of channels for multi-channel processing.
    parameters_init_method : str
        Initialization method for parameters.
    device : str
        Device to host the head (e.g., 'cpu' or 'cuda').

    Methods
    -------
    __init__(...)
        Initializes the PGM head with specified configurations.
    """
    def __init__(
        self, m: int, n: int,
        name: str = 'perceptron_head',
        distribution: str = 'normal',
        d: int = 2, with_replacement: bool = False,
        enable_bias: bool = False,
        # optional parameters
        with_lorr: bool = False,
        r: int = 3,
        with_residual: bool = False,
        channel_num: int = 1,
        # other parameters
        parameters_init_method: str = 'xavier_normal',
        device: str = 'cpu', *args, **kwargs
    ):
        """
        Initializes the PGM head.

        Parameters
        ----------
        m : int
            Input dimension.
        n : int
            Output dimension.
        name : str, optional
            Name of the PGM head, default is 'pgm_head'.
        distribution : str, optional
            Distribution type for combinatorial expansion, default is 'normal'.
        d : int, optional
            Degree for combinatorial expansion, default is 2.
        with_replacement : bool, optional
            Whether combinations are generated with replacement, default is False.
        enable_bias : bool, optional
            Whether to enable bias in reconciliation functions, default is False.
        with_lorr : bool, optional
            Whether to use LORR reconciliation, default is False.
        r : int, optional
            Parameter for reconciliation functions, default is 3.
        with_residual : bool, optional
            Whether to include a residual connection, default is False.
        channel_num : int, optional
            Number of channels for multi-channel processing, default is 1.
        parameters_init_method : str, optional
            Initialization method for parameters, default is 'xavier_normal'.
        device : str, optional
            Device to host the head, default is 'cpu'.

        Returns
        -------
        None
        """
        if distribution == 'normal':
            data_transformation = combinatorial_normal_expansion(
                d=d, with_replacement=with_replacement,
                device=device,
            )
        else:
            raise ValueError('tinybig only supports normal, exponential, cauchy, gamma, laplace or chi2 distributions...')

        if with_lorr:
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
            data_transformation=data_transformation,
            parameter_fabrication=parameter_fabrication,
            remainder=remainder,
            channel_num=channel_num,
            parameters_init_method=parameters_init_method,
            device=device, *args, **kwargs
        )

