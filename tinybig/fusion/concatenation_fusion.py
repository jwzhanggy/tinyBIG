# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

########################################
# Concatenation based Fusion Functions #
########################################

"""
The concatenation based fusion functions

This module contains the concatenation_fusion function.
"""

import torch
from tinybig.fusion import fusion


class concatenation_fusion(fusion):
    r"""
        A fusion mechanism that concatenates input tensors along their last dimension.

        Notes
        ----------

        Formally, given input interdependence matrices $\mathbf{A}_1, \mathbf{A}_2, \ldots, \mathbf{A}_k$, where each matrix $\mathbf{A}_i \in R^{m \times n_i}$ has $m$ rows and $n_i$ columns, we define the fusion operator as follows:

        $$
            \begin{equation}
            \begin{aligned}
            \mathbf{A} &= \text{fusion}(\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k) \\
            &= \left( \mathbf{A}_1 \sqcup \mathbf{A}_2 \sqcup \cdots \sqcup \mathbf{A}_k \right) \in R^{m \times n}
            \end{aligned}
            \end{equation}
        $$

        where $\sqcup$ denotes the row-wise concatenation of the matrices. The concatenation of these interdependence matrices results in a relatively large dimension,
        specifically $\sqcup_{i=1}^k \mathbf{A}_i \in R^{m \times (\sum_{i=1}^k n_i)}$, i.e., $n = \sum_{i=1}^k n_i$.

        Attributes
        ----------
        dims : list[int] | tuple[int]
            List or tuple specifying the dimensions of the input tensors.
        require_parameters : bool
            Indicates whether the fusion requires learnable parameters. Defaults to False.

        Methods
        -------
        calculate_n(dims=None, *args, **kwargs)
            Computes the total output dimension after concatenation.
        calculate_l(*args, **kwargs)
            Computes the number of learnable parameters (always 0 for concatenation).
        forward(x, w=None, device='cpu', *args, **kwargs)
            Performs concatenation fusion on the input tensors.
        """

    def __init__(self, dims: list[int] | tuple[int] = None, name: str = "concatenation_fusion", require_parameters: bool = False, *args, **kwargs):
        """
            Initializes the concatenation fusion function.

            Parameters
            ----------
            dims : list[int] | tuple[int], optional
                List or tuple specifying the dimensions of the input tensors. Defaults to None.
            name : str, optional
                Name of the fusion function. Defaults to "concatenation_fusion".
            require_parameters : bool, optional
                Indicates whether the fusion requires learnable parameters. Defaults to False.
            *args : tuple
                Additional positional arguments for the parent class.
            **kwargs : dict
                Additional keyword arguments for the parent class.
        """
        super().__init__(dims=dims, name=name, require_parameters=False, *args, **kwargs)

    def calculate_n(self, dims: list[int] | tuple[int] = None, *args, **kwargs):
        """
            Computes the total output dimension after concatenation.

            Parameters
            ----------
            dims : list[int] | tuple[int], optional
                List or tuple specifying the dimensions of the input tensors. Defaults to None.

            Returns
            -------
            int
                Total output dimension after concatenation.

            Raises
            ------
            AssertionError
                If the `dims` argument is not provided and the instance does not have `dims`.
        """
        dims = dims if dims is not None else self.dims
        assert dims is not None
        return sum(dims)

    def calculate_l(self, *args, **kwargs):
        """
            Computes the number of learnable parameters.

            Returns
            -------
            int
                Always returns 0, as concatenation does not involve learnable parameters.
        """
        return 0

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor], w: torch.nn.Parameter = None, device: str = 'cpu', *args, **kwargs):
        """
            Performs concatenation fusion on the input tensors.

            Parameters
            ----------
            x : list[torch.Tensor] | tuple[torch.Tensor]
                List or tuple of input tensors to be concatenated.
            w : torch.nn.Parameter, optional
                Parameter tensor (unused for concatenation). Defaults to None.
            device : str, optional
                Device for computation ('cpu', 'cuda'). Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                Fused tensor after concatenation.

            Raises
            ------
            ValueError
                If `x` is empty or if input tensors have inconsistent shapes, excluding the last dimension.
            AssertionError
                If the output dimension after concatenation does not match the calculated dimension.
        """

        if not x:
            raise ValueError("The input x cannot be empty...")
        if not all(x[0].shape[:-1] == t.shape[:-1] for t in x):
            raise ValueError("Excluding the last dimension, the input x contains elements of different shapes for other dimensions...")

        if all(x[0].shape == t.shape for t in x):
            # if they are all the same shape, it will allow some cross-channel pre-processing operators...
            x = torch.stack(x, dim=0)
            x = self.pre_process(x=x, device=device)
            x = [t.squeeze(dim=0) for t in x.split(1, dim=0)]
        else:
            # otherwise, we cannot perform cross channel preprocessing, and have to pre-process them individually...
            x = [self.pre_process(t, device=device) for t in x]

        fused_x = torch.cat(x, dim=-1)

        assert fused_x.size(-1) == self.calculate_n([element.size(-1) for element in x])
        return self.post_process(x=fused_x, device=device)
