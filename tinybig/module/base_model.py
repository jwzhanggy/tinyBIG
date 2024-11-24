# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
The base model template.

It defines the base model template for implementing the RPN models.
"""

from abc import abstractmethod

import torch
from torch.nn import Module

from tinybig.module.base_function import function
from tinybig.util.utility import create_directory_if_not_exists


class model(Module, function):
    r"""
    The base model class of the RPN model in the tinyBIG toolkit.

    It inherits from the torch.nn.Module class, which also inherits the
    "state_dict" and "load_state_dict" methods from the base class.

    ...

    Notes
    ---------

    ## RPN Model Architecture

    Formally, given the underlying data distribution mapping $f: {R}^m \to {R}^n$,
    the RPN model proposes to approximate function $f$ as follows:
    $$
        \begin{equation}
            g(\mathbf{x} | \mathbf{w}) = \left \langle \kappa_{\xi} (\mathbf{x}), \psi(\mathbf{w}) \right \rangle + \pi(\mathbf{x}),
        \end{equation}
    $$

    The RPN model disentangles input data from model parameters through the expansion functions $\kappa$ and
    reconciliation function $\psi$, subsequently summed with the remainder function $\pi$, where

    * $\kappa_{\xi}: {R}^m \to {R}^{D}$ is named as the **data interdependent transformation function**. It is a composite function of the **data transformation function** $\kappa$ and the **data interdependence function** $\xi$. Notation $D$ is the target expansion space dimension.

    * $\psi: {R}^l \to {R}^{n \times D}$ is named as the **parameter reconciliation function** (or **parameter fabrication function** to be general), which is defined only on the parameters without any input data.

    * $\pi: {R}^m \to {R}^n$ is named as the **remainder function**.

    * $\xi_a: {R}^{b \times m} \to {R}^{m \times m'}$ and $\xi_i: {R}^{b \times m} \to {R}^{b \times b'}$ defined on the input data batch $\mathbf{X} \in R^{b \times m}$ are named as the **attribute** and **instance data interdependence functions**, respectively.

    ## Deep RPN model with Multi-Layer

    The multi-head multi-channel RPN layer provides RPN with greater capabilities
    for approximating functions with diverse expansions concurrently.
    However, such shallow architectures can be insufficient for modeling complex functions.
    The RPN model can also be designed with a deep architecture by stacking multiple RPN layers on top of each other.

    Formally, we can represent the deep RPN model with multi-layers as follows:

    $$
        \begin{equation}
            \begin{cases}
                \text{Input: } & \mathbf{H}_0  = \mathbf{X},\\\\
                \text{Layer 1: } & \mathbf{H}_1 = \left\langle \kappa_{\xi, 1}(\mathbf{H}_0), \psi_1(\mathbf{w}_1) \right\rangle + \pi_1(\mathbf{H}_0),\\\\
                \text{Layer 2: } & \mathbf{H}_2 = \left\langle \kappa_{\xi, 2}(\mathbf{H}_1), \psi_2(\mathbf{w}_2) \right\rangle + \pi_2(\mathbf{H}_1),\\\\
                \cdots & \cdots \ \cdots\\\\
                \text{Layer K: } & \mathbf{H}_K = \left\langle \kappa_{\xi, K}(\mathbf{H}_{K-1}), \psi_K(\mathbf{w}_K) \right\rangle + \pi_K(\mathbf{H}_{K-1}),\\\\
                \text{Output: } & \mathbf{Z}  = \mathbf{H}_K.
            \end{cases}
        \end{equation}
    $$

    In the above equation, the subscripts used above denote the layer index. The dimensions of the outputs at each layer
    can be represented as a list $[d_0, d_1, \cdots, d_{K-1}, d_K]$, where $d_0 = m$ and $d_K = n$
    denote the input and the desired output dimensions, respectively.
    Therefore, if the component functions at each layer of our model have been predetermined, we can just use the dimension
    list $[d_0, d_1, \cdots, d_{K-1}, d_K]$ to represent the architecture of the RPN model.

    Attributes
    ----------
    name: str, default = 'base_metric'
        Name of the model.

    Methods
    ----------
    __init__
        It performs the initialization of the model

    save_ckpt
        It saves the model state as checkpoint to file.

    load_ckpt
        It loads the model state from a file.

    __call__
        It reimplementation the build-in callable method.

    forward
        The forward method of the model.
    """
    def __init__(self, name: str = 'model_name', device: str = 'cpu', *args, **kwargs):
        """
        The initialization method of the base model class.

        It initializes a model object based on the provided model parameters.

        Parameters
        ----------
        name: str, default = 'model_name'
            The name of the model, with default value "model_name".

        Returns
        ----------
        object
            The initialized model object.
        """
        Module.__init__(self)
        function.__init__(self, name=name, device=device)

    def save_ckpt(self, cache_dir='./ckpt', checkpoint_file='checkpoint'):
        """
        The model state checkpoint saving method.

        It saves the current model state to a checkpoint file.

        Parameters
        ----------
        cache_dir: str, default = './ckpt'
            The cache directory of the model checkpoint file.
        checkpoint_file: str, default = 'checkpoint'
            The checkpoint file name.

        Returns
        -------
        None
            This method doesn't have return values.
        """
        create_directory_if_not_exists(f'{cache_dir}/{checkpoint_file}')
        torch.save(self.state_dict(), f'{cache_dir}/{checkpoint_file}')
        print("model checkpoint saving to {}/{}...".format(cache_dir, checkpoint_file))

    def load_ckpt(self, cache_dir: str = './ckpt', checkpoint_file: str = 'checkpoint', strict: bool = True):
        """
        The model state checkpoint loading method.

        It loads the model state from the provided checkpoint file.

        Parameters
        ----------
        cache_dir: str, default = './ckpt'
            The cache directory of the model checkpoint file.
        checkpoint_file: str, default = 'checkpoint'
            The checkpoint file name.
        strict: bool, default = True
            The boolean tag of whether the model state loading follows the strict configuration checking.

        Returns
        -------
        None
            This method doesn't have return values.
        """
        self.load_state_dict(torch.load(f'{cache_dir}/{checkpoint_file}'), strict=strict)
        print("model checkpoint loading from {}/{}...".format(cache_dir, checkpoint_file))

    @abstractmethod
    def to_config(self, *args, **kwargs):
        """
        Abstract method to convert the `model` instance into a configuration dictionary.

        This method is intended to be implemented by subclasses. It should generate a dictionary
        that encapsulates the essential configuration of the model, allowing for reconstruction
        or serialization of the instance. The specific structure and content of the configuration
        dictionary are determined by the implementing model.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments that might be required by the implementation.
        **kwargs : dict
            Additional keyword arguments that might be required by the implementation.

        Returns
        -------
        dict
            A dictionary representing the configuration of the instance. The exact structure and keys
            depend on the subclass implementation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass and is called directly.

        See Also
        --------
        BaseClass : The base class where this method is defined.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        The forward method of the model.

        It is declared to be an abstractmethod and needs to be implemented in the inherited RPN model classes.
        This callable method accepts the data instances as the input and generate the desired outputs.

        Returns
        ----------
        torch.Tensor
            The model generated outputs.
        """
        pass