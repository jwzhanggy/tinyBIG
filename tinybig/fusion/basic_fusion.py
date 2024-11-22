# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Basic Fusion Functions #
##########################

"""
The basic fusion functions

This module contains the basic fusion function, such as
    weighted_summation_fusion,
    summation_fusion,
    average_fusion,
    average_fusion as mean_fusion,
    parameterized_weighted_summation_fusion.
"""

import torch

from tinybig.fusion import fusion


class weighted_summation_fusion(fusion):
    r"""
        A fusion mechanism that combines inputs using a weighted summation.

        Notes
        ----------

        Formally, given interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k \in R^{m \times n}$ of dimension $m \times n$, we can combine them through a weighted summation as follows:

        $$
            \begin{equation}
            \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) = \sum_{i=1}^k \alpha_i \mathbf{A}_i \in R^{m \times n},
            \end{equation}
        $$

        where $\alpha_i$ represents the weight assigned to matrix $\mathbf{A}_i$ for each $i \in \{1, 2, \cdots, k\}$.

        Attributes
        ----------
        weights : torch.Tensor
            Tensor containing weights for the summation. If `require_parameters` is True, weights are learned.

        Methods
        -------
        calculate_n(dims=None, *args, **kwargs)
            Computes the output dimension of the fused input.
        calculate_l(*args, **kwargs)
            Computes the number of learnable parameters, if applicable.
        forward(x, w=None, device='cpu', *args, **kwargs)
            Performs the weighted summation fusion on the input tensors.
    """

    def __init__(self, dims: list[int] | tuple[int] = None, weights: torch.Tensor = None, name: str = "weighted_summation_fusion", *args, **kwargs):
        """
            Initializes the weighted summation fusion function.

            Parameters
            ----------
            dims : list[int] | tuple[int], optional
                Dimensions of the input tensors. Defaults to None.
            weights : torch.Tensor, optional
                Predefined weights for the summation. Defaults to None.
            name : str, optional
                Name of the fusion function. Defaults to "weighted_summation_fusion".
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(dims=dims, name=name, *args, **kwargs)
        self.weights = weights

    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        """
            Computes the output dimension of the fused input.

            Parameters
            ----------
            dims : list[int] | tuple[int], optional
                List of dimensions of the input tensors. Defaults to None.

            Returns
            -------
            int
                Output dimension, equal to the input dimension if consistent.

            Raises
            ------
            AssertionError
                If input dimensions are inconsistent.
        """
        dims = dims if dims is not None else self.dims
        assert dims is not None and all(dim == dims[0] for dim in dims)
        return dims[0]

    def calculate_l(self, *args, **kwargs):
        """
            Computes the number of learnable parameters, if applicable.

            Returns
            -------
            int
                Number of learnable parameters. Returns 0 if `require_parameters` is False.
        """
        if self.require_parameters:
            return self.get_num()
        else:
            return 0

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor], w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Performs the weighted summation fusion on the input tensors.

            Parameters
            ----------
            x : list[torch.Tensor] | tuple[torch.Tensor]
                List or tuple of input tensors to be fused.
            w : torch.nn.Parameter, optional
                Learnable weights for fusion. Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Fused tensor after weighted summation.

            Raises
            ------
            ValueError
                If `x` is empty or if input tensors have inconsistent shapes.
            AssertionError
                If weights are not provided or have inconsistent dimensions with inputs.
        """

        if not x:
            raise ValueError("The input x cannot be empty...")
        if not all(x[0].shape == t.shape for t in x):
            raise ValueError("The input x must have the same shape.")

        x = torch.stack(x)
        x = self.pre_process(x=x, device=device)

        if self.require_parameters:
            weights = w
        else:
            weights = self.weights

        assert x.ndim >= 1 and all(item.shape == x[0].shape for item in x)
        assert weights is not None and x.size(0) == weights.size(0)

        weights = weights.view(-1, *[1]*(len(x[0].shape)))
        fused_x = (x * weights).sum(dim=0)

        assert fused_x.size(-1) == self.calculate_n([element.size(-1) for element in x])
        return self.post_process(x=fused_x, device=device)


class summation_fusion(weighted_summation_fusion):
    """
        A fusion mechanism that combines inputs using simple summation.

        It inherits from the weighted summation fusion class, and the inputs are treated with equal importance with weight value $1$.

        Attributes
        ----------
        weights : torch.Tensor
            Predefined weights, initialized to 1 for all inputs.

        Methods
        -------
        __init__(...)
            Initializes the summation fusion function.
    """
    def __init__(self, dims: list[int] | tuple[int], name: str = "summation_fusion", require_parameters: bool = False,  *args, **kwargs):
        """
            Initializes the summation fusion function.

            Parameters
            ----------
            dims : list[int] | tuple[int]
                Dimensions of the input tensors.
            name : str, optional
                Name of the fusion function. Defaults to "summation_fusion".
            require_parameters : bool, optional
                Whether parameters are required. Defaults to False.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(dims=dims, weights=torch.ones(len(dims)), name=name, require_parameters=False, *args, **kwargs)


class average_fusion(weighted_summation_fusion):
    """
        A fusion mechanism that combines inputs using simple averaging.

        It inherits from the weighted summation fusion class, and the $k$ inputs are treated with equal importance with weight value $1/k$.

        Attributes
        ----------
        weights : torch.Tensor
            Predefined weights, initialized to 1/N for all inputs, where N is the number of inputs.

        Methods
        -------
        __init__(...)
            Initializes the average fusion function.
    """
    def __init__(self, dims: list[int] | tuple[int], name: str = "average_fusion", require_parameters: bool = False, *args, **kwargs):
        """
            Initializes the average fusion function.

            Parameters
            ----------
            dims : list[int] | tuple[int]
                Dimensions of the input tensors.
            name : str, optional
                Name of the fusion function. Defaults to "average_fusion".
            require_parameters : bool, optional
                Whether parameters are required. Defaults to False.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(dims=dims, weights=1.0/len(dims)*torch.ones(len(dims)), name=name, require_parameters=False, *args, **kwargs)


class parameterized_weighted_summation_fusion(weighted_summation_fusion):
    """
        A fusion mechanism that combines inputs using a parameterized weighted summation.

        It inherits from the weighted summation fusion class, and the inputs are treated with a learnable parameter denoting their importance weights.

        Attributes
        ----------
        weights : None
            Weights are learnable parameters in this class.

        Methods
        -------
        __init__(...)
            Initializes the parameterized weighted summation fusion function.
    """
    def __init__(self, dims: list[int] | tuple[int] = None, name: str = "parameterized_weighted_summation_fusion", require_parameters: bool = True, *args, **kwargs):
        """
            Initializes the parameterized weighted summation fusion function.

            Parameters
            ----------
            dims : list[int] | tuple[int], optional
                Dimensions of the input tensors. Defaults to None.
            name : str, optional
                Name of the fusion function. Defaults to "parameterized_weighted_summation_fusion".
            require_parameters : bool, optional
                Whether parameters are required. Defaults to True.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(dims=dims, weights=None, name=name, require_parameters=True, *args, **kwargs)


