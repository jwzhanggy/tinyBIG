# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
Deep RPN model with multiple RPN layers.

This module contains the implementation of the RPN model composed of multiple RPN layers.

"""
import torch

from tinybig.util.util import get_obj_from_str, process_num_alloc_configs
from tinybig.model.base_model import model


class rpn(model):
    r"""
    The RPN model build in the tinyBIG toolkit.

    It implements the RPN model build with a deep architecture involving multiple RPN layers.

    Notes
    ----------
    The RPN model can be built with a deep architecture by stacking multiple RPN layers on top of each other.
    A deep RPN model with K layers can be represented as follows:
    $$
        \begin{equation}
            \begin{cases}
                \text{Input: } & \mathbf{h}_0  = \mathbf{x},\\\\
                \text{Layer 1: } & \mathbf{h}\_1 = \left\langle \kappa\_1(\mathbf{h}\_0), \psi\_1(\mathbf{w}\_1) \right\rangle + \pi\_1(\mathbf{h}\_0),\\\\
                \text{Layer 2: } & \mathbf{h}\_2 = \left\langle \kappa\_2(\mathbf{h}\_1), \psi\_2(\mathbf{w}\_2) \right\rangle + \pi\_2(\mathbf{h}\_1),\\\\
                \cdots & \cdots \ \cdots\\\\
                \text{Layer K: } & \mathbf{h}\_K = \left\langle \kappa\_K(\mathbf{h}\_{K-1}), \psi\_K(\mathbf{w}\_K) \right\rangle + \pi\_K(\mathbf{h}\_{K-1}),\\\\
                \text{Output: } & \hat{\mathbf{y}}  = \mathbf{h}_K.
            \end{cases}
        \end{equation}
    $$
    Each of the layers shown above can be implemented with one RPN layer, which may also involve multi-heads and multi-channels.

    Attributes
    ----------
    name: str, default = 'Reconciled_Polynomial_Network'
        Name of the RPN model.
    layers: list, default = None
        The model architecture with multiple layers.
    device: str, default = 'cpu'
        Device to host the RPN model.

    Methods
    ----------
    __init__
        The RPN model initialization method, it will initialize the model architecture based on the input configurations.
    get_depthber
        The RPN model depth retrieval method, it will get the number of layers involved in the RPN model.
    initialize_parameters
        The RPN model parameter initialization method, it will initialize the parameters of each layer.
    forward
        The forward method of the RPN model. It will generate the outputs based on the input data.
    """
    def __init__(
            self,
            name='Reconciled_Polynomial_Network',
            layers: list = None,
            layer_configs: dict | list = None,
            depth: int = None,
            depth_alloc: int | list = None,
            device='cpu',
            *args, **kwargs
    ):
        """
        The initialization method of the RPN model with multiple RPN layers.

        It initializes the deep RPN model composed with multiple RPN layers.
        Specifically, this method initializes the name, layers and devices to host the RPN model.

        Parameters
        ----------
        name: str, default = 'Reconciled_Polynomial_Network'
            Name of the RPN model.
        layers: list, default = None
            The list of RPN layers in the model. The layers involved in the model can be initialized
            either directly with the layers parameter or via the layer_configs parameter.
        layer_configs: list, default = None
            The list of RPN layer detailed configurations.
        depth: int, default = None
            The total layer number of the model. It is optional, if the "layers" or the "layer_configs" can provide
            sufficient information for the model initialization, this depth parameter can be set as None.
        depth_alloc: list, default = None
            RPN allows the layers with different configurations, instead of listing such configurations one by one,
            it also allows the listing of each configuration types together with the repeating numbers for
            each of the layers, which are specified by this optional layer allocation parameter.
        device: str, default = 'cpu'
            The device for hosting the RPN layer.

        Returns
        ----------
        object
            The initialized RPN model object.
        """
        # initialize the base model class with the name
        super().__init__(name=name, *args, **kwargs)

        # the model layers are initialized as an empty modulelist
        self.layers = torch.nn.ModuleList()
        self.device = device

        if layers is not None:
            # initialize layers from the layers parameter directly
            self.layers.extend(layers)
            depth = len(self.layers)
        elif layer_configs is None:
            raise ValueError("Both layers and layer_configs are None, the model cannot be initialized...")
        else:
            # initialize layers from the layer configs
            depth, depth_alloc, layer_configs = process_num_alloc_configs(depth, depth_alloc, layer_configs)
            assert len(depth_alloc) == len(layer_configs) and depth == sum(depth_alloc)

            for layer_repeat_time, layer_config in zip(depth_alloc, layer_configs):
                for layer_id in range(layer_repeat_time):
                    class_name = layer_config['layer_class']
                    parameters = layer_config['layer_parameters'] if 'layer_parameters' in layer_config else {}
                    parameters['device'] = device
                    layer = get_obj_from_str(class_name)(**parameters)
                    self.layers.append(layer)
        assert len(self.layers) == depth
        # initialize the model parameters
        self.initialize_parameters()

    def get_depth(self):
        """
        RPN model name retrieval method.

        It returns the name of the RPN model.

        Returns
        -------
        str
            It returns the name of the RPN model.
        """
        return len(self.layers)

    def initialize_parameters(self):
        """
        RPN model parameter initialization method.

        It initializes the parameters of the RPN model with deep architectures.
        This method will call the "initialize_parameters" method for each of the involved layers.

        Returns
        -------
        None
            This method doesn't have any return values.
        """
        for layer in self.layers:
            layer.initialize_parameters()

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        r"""
        The forward method of the RPN model.

        It computes the desired outputs based on the data inputs.
        For the RPN with deep architecture involving multiple layers, this method will iteratively call each of the
        layers to process the data inputs, illustrated as follows:

        $$
            \begin{equation}
                \begin{cases}
                    \text{Input: } & \mathbf{h}_0  = \mathbf{x},\\\\
                    \text{Layer 1: } & \mathbf{h}_1 = \left\langle \kappa_1(\mathbf{h}_0), \psi_1(\mathbf{w}_1) \right\rangle + \pi_1(\mathbf{h}_0),\\\\
                    \text{Layer 2: } & \mathbf{h}_2 = \left\langle \kappa_2(\mathbf{h}_1), \psi_2(\mathbf{w}_2) \right\rangle + \pi_2(\mathbf{h}_1),\\\\
                    \cdots & \cdots \ \cdots\\\\
                    \text{Layer K: } & \mathbf{h}_K = \left\langle \kappa_K(\mathbf{h}_{K-1}), \psi_K(\mathbf{w}_K) \right\rangle + \pi_K(\mathbf{h}_{K-1}),\\\\
                    \text{Output: } & \hat{\mathbf{y}}  = \mathbf{h}_K.
                \end{cases}
            \end{equation}
        $$


        Parameters
        ----------
        x: torch.Tensor
            The input data instances.
        device: str, default = 'cpu'
            Device for processing the data inputs.

        Returns
        -------
        torch.Tensor
            The desired outputs generated by the RPN model for the input data instances.
        """
        for layer in self.layers:
            x = layer(x, device=device)
        return x

