# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#####################################
# Wavelet based Expansion Functions #
#####################################

"""
The wavelet based data expansion functions.

This module contains the orthogonal polynomial data expansion functions, including
    meyer_wavelet_expansion,
    ricker_wavelet_expansion,
    shannon_wavelet_expansion,
    beta_wavelet_expansion,
    harr_wavelet_expansion,
    dog_wavelet_expansion
"""

import itertools
import numpy as np

import torch.nn

from tinybig.expansion import transformation
from tinybig.koala.signal_processing import (
    discrete_wavelet,
    harr_wavelet,
    dog_wavelet,
    beta_wavelet,
    ricker_wavelet,
    shannon_wavelet,
    meyer_wavelet
)


class discrete_wavelet_expansion(transformation):
    r"""
        Discrete Wavelet Expansion Transformation.

        Implements the discrete wavelet expansion transformation, enabling feature expansion based on wavelet functions.

        Notes
        ---------

        Formally, given the input variable $\mathbf{x} \in R^{m}$, to approximate the underlying mapping $f: R^m \to R^n$ with wavelet analysis, we can define the approximated output as

        $$
            \begin{equation}
            f(\mathbf{x}) \approx \sum_{s, t} \left \langle f(\mathbf{x}), \phi_{s, t} (\mathbf{x} | a, b) \right \rangle \cdot \phi_{s, t} (\mathbf{x} | a, b),
            \end{equation}
        $$

        where $\phi_{s, t} (\cdot | a, b)$ denotes the child wavelet defined by hyper-parameters $a > 1$ and $b > 0$:

        $$
            \begin{equation}
            \phi_{s, t}(x | a, b) = \frac{1}{\sqrt{a^s}} \phi \left( \frac{x - t \cdot b \cdot a^s}{a^s} \right).
            \end{equation}
        $$

        Based on the wavelet mapping $\phi_{s, t} (\cdot | a, b)$, we can introduce the $1_{st}$-order and $2_{nd}$-order wavelet data expansion functions as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=1) = \left[ \phi_{0, 0}(\mathbf{x}), \phi_{0, 1}(\mathbf{x}), \cdots, \phi_{s, t}(\mathbf{x}) \right] \in R^{D_1}.
            \end{equation}
        $$

        and

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=2) = \kappa(\mathbf{x} | d=1) \otimes \kappa(\mathbf{x} | d=1) \in R^{D_2}.
            \end{equation}
        $$

        The output dimensions of the order-1 and order-2 wavelet expansions are $D_1 = s \cdot t \cdot m$ and $D_2 = (s \cdot t \cdot m)^2$, respectively.


        Attributes
        ----------
        name : str
            Name of the transformation.
        d : int
            Maximum order of wavelet-based polynomial expansion.
        s : int
            Number of scaling factors for the wavelet.
        t : int
            Number of translation factors for the wavelet.
        wavelet : callable
            The wavelet function applied during the transformation.

        Methods
        -------
        calculate_D(m: int)
            Calculate the total dimensionality of the expanded feature space.
        wavelet_x(x: torch.Tensor, device: str = 'cpu', *args, **kwargs)
            Apply the wavelet transformation to the input data.
        forward(x: torch.Tensor, device: str = 'cpu', *args, **kwargs)
            Perform the discrete wavelet expansion on the input data.
    """
    def __init__(self, name: str = 'discrete_wavelet_expansion', d: int = 1, s: int = 1, t: int = 1, *args, **kwargs):
        """
            Initializes the discrete wavelet expansion transformation.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'discrete_wavelet_expansion'.
            d : int, optional
                The maximum order of wavelet-based polynomial expansion. Defaults to 1.
            s : int, optional
                The number of scaling factors for the wavelet. Defaults to 1.
            t : int, optional
                The number of translation factors for the wavelet. Defaults to 1.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.d = d
        self.s = s
        self.t = t
        self.wavelet = None

    def calculate_D(self, m: int):
        """
            Calculates the expanded dimensionality of the transformed data.

            Parameters
            ----------
            m : int
                The original number of features.

            Returns
            -------
            int
                The total number of features after expansion.
        """
        return np.sum([(m * self.s * self.t) ** d for d in range(1, self.d + 1)])

    def wavelet_x(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Applies the wavelet function to the input data.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor to be transformed.
            device : str, optional
                The device to perform computation on. Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The transformed tensor after applying the wavelet function.
        """
        assert self.wavelet is not None and isinstance(self.wavelet, discrete_wavelet)

        combinations = list(itertools.product(range(self.s), range(self.t)))
        combination_index = {comb: idx for idx, comb in enumerate(combinations)}

        expansion = torch.ones(size=[x.size(0), x.size(1), self.s * self.t]).to(device)
        for s, t in combination_index:
            n = combination_index[(s, t)]
            expansion[:, :, n] = self.wavelet(x=x, s=s, t=t)
        expansion = expansion[:, :, :].contiguous().view(x.size(0), -1)
        return expansion

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        """
            Expands the input data using discrete wavelet expansion.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor to be expanded.
            device : str, optional
                The device to perform computation on. Defaults to 'cpu'.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.

            Returns
            -------
            torch.Tensor
                The expanded tensor.
        """
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        wavelet_x = self.wavelet_x(x, device=device, *args, **kwargs)

        if self.d > 1:
            wavelet_x_powers = torch.ones(size=[wavelet_x.size(0), 1]).to(device)
            expansion = torch.Tensor([]).to(device)

            for i in range(1, self.d + 1):
                wavelet_x_powers = torch.einsum('ba,bc->bac', wavelet_x_powers.clone(), wavelet_x).view(wavelet_x_powers.size(0), wavelet_x_powers.size(1) * wavelet_x.size(1))
                expansion = torch.cat((expansion, wavelet_x_powers), dim=1)
        else:
            expansion = wavelet_x

        assert expansion.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=expansion, device=device)


class harr_wavelet_expansion(discrete_wavelet_expansion):
    r"""
        Haar Wavelet Expansion.

        Applies the Haar wavelet function for feature expansion.

        Notes
        ----------
        Formally, given the input variable $\mathbf{x} \in R^{m}$, to approximate the underlying mapping $f: R^m \to R^n$ with wavelet analysis, we can define the approximated output as

        $$
            \begin{equation}
            f(\mathbf{x}) \approx \sum_{s, t} \left \langle f(\mathbf{x}), \phi_{s, t} (\mathbf{x} | a, b) \right \rangle \cdot \phi_{s, t} (\mathbf{x} | a, b),
            \end{equation}
        $$

        where $\phi_{s, t} (\cdot | a, b)$ denotes the child wavelet defined by hyper-parameters $a > 1$ and $b > 0$:

        $$
            \begin{equation}
            \phi_{s, t}(x | a, b) = \frac{1}{\sqrt{a^s}} \phi \left( \frac{x - t \cdot b \cdot a^s}{a^s} \right).
            \end{equation}
        $$

        Based on the wavelet mapping $\phi_{s, t} (\cdot | a, b)$, we can introduce the $1_{st}$-order and $2_{nd}$-order wavelet data expansion functions as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=1) = \left[ \phi_{0, 0}(\mathbf{x}), \phi_{0, 1}(\mathbf{x}), \cdots, \phi_{s, t}(\mathbf{x}) \right] \in R^{D_1}.
            \end{equation}
        $$

        and

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=2) = \kappa(\mathbf{x} | d=1) \otimes \kappa(\mathbf{x} | d=1) \in R^{D_2}.
            \end{equation}
        $$

        The output dimensions of the order-1 and order-2 wavelet expansions are $D_1 = s \cdot t \cdot m$ and $D_2 = (s \cdot t \cdot m)^2$, respectively.

        Specifically, the functions $\left\{ \phi_{s, t}\right\}_{ s, t \in Z}$ defines the orthonormal basis of the space and
        the mapping $\phi(\cdot)$ used in the child wavelet may have different representations.

        For Harr wavelet, it can be represented as follows:

        __Harr Wavelet:__

        $$
            \begin{equation}
            \begin{aligned}
            &\phi(\tau) = \begin{cases}
            1, & 0 \le \tau < \frac{1}{2},\\
            -1, & \frac{1}{2} \le \tau < 1,\\
            0, & \text{ otherwise}.
            \end{cases}
            \end{aligned}
            \end{equation}
        $$


        Attributes
        ----------
        wavelet : callable
            Haar wavelet function used during the transformation.

        Methods
        -------
        Inherits all methods from `discrete_wavelet_expansion`.
    """
    def __init__(self, name: str = 'harr_wavelet_expansion', a: float = 1.0, b: float = 1.0, *args, **kwargs):
        """
            Initializes the Haar wavelet expansion.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'harr_wavelet_expansion'.
            a : float, optional
                The scaling factor for the wavelet. Defaults to 1.0.
            b : float, optional
                The translation factor for the wavelet. Defaults to 1.0.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = harr_wavelet(a=a, b=b)


class beta_wavelet_expansion(discrete_wavelet_expansion):
    r"""
        Beta Wavelet Expansion.

        Applies the Beta wavelet function for feature expansion.

        Notes
        ----------
        Formally, given the input variable $\mathbf{x} \in R^{m}$, to approximate the underlying mapping $f: R^m \to R^n$ with wavelet analysis, we can define the approximated output as

        $$
            \begin{equation}
            f(\mathbf{x}) \approx \sum_{s, t} \left \langle f(\mathbf{x}), \phi_{s, t} (\mathbf{x} | a, b) \right \rangle \cdot \phi_{s, t} (\mathbf{x} | a, b),
            \end{equation}
        $$

        where $\phi_{s, t} (\cdot | a, b)$ denotes the child wavelet defined by hyper-parameters $a > 1$ and $b > 0$:

        $$
            \begin{equation}
            \phi_{s, t}(x | a, b) = \frac{1}{\sqrt{a^s}} \phi \left( \frac{x - t \cdot b \cdot a^s}{a^s} \right).
            \end{equation}
        $$

        Based on the wavelet mapping $\phi_{s, t} (\cdot | a, b)$, we can introduce the $1_{st}$-order and $2_{nd}$-order wavelet data expansion functions as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=1) = \left[ \phi_{0, 0}(\mathbf{x}), \phi_{0, 1}(\mathbf{x}), \cdots, \phi_{s, t}(\mathbf{x}) \right] \in R^{D_1}.
            \end{equation}
        $$

        and

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=2) = \kappa(\mathbf{x} | d=1) \otimes \kappa(\mathbf{x} | d=1) \in R^{D_2}.
            \end{equation}
        $$

        The output dimensions of the order-1 and order-2 wavelet expansions are $D_1 = s \cdot t \cdot m$ and $D_2 = (s \cdot t \cdot m)^2$, respectively.

        Specifically, the functions $\left\{ \phi_{s, t}\right\}_{ s, t \in Z}$ defines the orthonormal basis of the space and
        the mapping $\phi(\cdot)$ used in the child wavelet may have different representations.

        For Beta wavelet, it can be represented as follows:

        __Beta Wavelet:__

        $$
            \begin{equation}
            \begin{aligned}
            &\phi(\tau | \alpha, \beta) = \frac{1}{B(\alpha, \beta)} \tau^{\alpha - 1} (1-\tau)^{\beta -1},
            \end{aligned}
            \end{equation}
        $$
        where $\alpha, \beta \in [1, \infty]$.

        Attributes
        ----------
        wavelet : callable
            Beta wavelet function used during the transformation.

        Methods
        -------
        Inherits all methods from `discrete_wavelet_expansion`.
    """
    def __init__(self, name: str = 'beta_wavelet_expansion', a: float = 1.0, b: float = 1.0, alpha: float = 1.0, beta: float = 1.0,  *args, **kwargs):
        """
            Initializes the Beta wavelet expansion.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'beta_wavelet_expansion'.
            a : float, optional
                The scaling factor for the wavelet. Defaults to 1.0.
            b : float, optional
                The translation factor for the wavelet. Defaults to 1.0.
            alpha : float, optional
                Alpha parameter for the Beta wavelet. Defaults to 1.0.
            beta : float, optional
                Beta parameter for the Beta wavelet. Defaults to 1.0.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = beta_wavelet(a=a, b=b, alpha=alpha, beta=beta)


class shannon_wavelet_expansion(discrete_wavelet_expansion):
    r"""
        Shannon Wavelet Expansion.

        Applies the Shannon wavelet function for feature expansion.

        Notes
        ----------
        Formally, given the input variable $\mathbf{x} \in R^{m}$, to approximate the underlying mapping $f: R^m \to R^n$ with wavelet analysis, we can define the approximated output as

        $$
            \begin{equation}
            f(\mathbf{x}) \approx \sum_{s, t} \left \langle f(\mathbf{x}), \phi_{s, t} (\mathbf{x} | a, b) \right \rangle \cdot \phi_{s, t} (\mathbf{x} | a, b),
            \end{equation}
        $$

        where $\phi_{s, t} (\cdot | a, b)$ denotes the child wavelet defined by hyper-parameters $a > 1$ and $b > 0$:

        $$
            \begin{equation}
            \phi_{s, t}(x | a, b) = \frac{1}{\sqrt{a^s}} \phi \left( \frac{x - t \cdot b \cdot a^s}{a^s} \right).
            \end{equation}
        $$

        Based on the wavelet mapping $\phi_{s, t} (\cdot | a, b)$, we can introduce the $1_{st}$-order and $2_{nd}$-order wavelet data expansion functions as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=1) = \left[ \phi_{0, 0}(\mathbf{x}), \phi_{0, 1}(\mathbf{x}), \cdots, \phi_{s, t}(\mathbf{x}) \right] \in R^{D_1}.
            \end{equation}
        $$

        and

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=2) = \kappa(\mathbf{x} | d=1) \otimes \kappa(\mathbf{x} | d=1) \in R^{D_2}.
            \end{equation}
        $$

        The output dimensions of the order-1 and order-2 wavelet expansions are $D_1 = s \cdot t \cdot m$ and $D_2 = (s \cdot t \cdot m)^2$, respectively.

        Specifically, the functions $\left\{ \phi_{s, t}\right\}_{ s, t \in Z}$ defines the orthonormal basis of the space and
        the mapping $\phi(\cdot)$ used in the child wavelet may have different representations.

        For Shannon wavelet, it can be represented as follows:

        __Shannon Wavelet:__

        $$
            \begin{equation}
            \begin{aligned}
            &\phi(\tau) = \frac{\sin(2\pi \tau) - \sin(\pi \tau)}{\pi \tau}.
            \end{aligned}
            \end{equation}
        $$

        Attributes
        ----------
        wavelet : callable
            Shannon wavelet function used during the transformation.

        Methods
        -------
        Inherits all methods from `discrete_wavelet_expansion`.
    """
    def __init__(self, name: str = 'shannon_wavelet_expansion', a: float = 1.0, b: float = 1.0, *args, **kwargs):
        """
            Initializes the Shannon wavelet expansion.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'shannon_wavelet_expansion'.
            a : float, optional
                The scaling factor for the wavelet. Defaults to 1.0.
            b : float, optional
                The translation factor for the wavelet. Defaults to 1.0.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = shannon_wavelet(a=a, b=b)


class ricker_wavelet_expansion(discrete_wavelet_expansion):
    r"""
        Ricker (Mexican Hat) Wavelet Expansion.

        Applies the Ricker wavelet function for feature expansion.

        Notes
        ----------
        Formally, given the input variable $\mathbf{x} \in R^{m}$, to approximate the underlying mapping $f: R^m \to R^n$ with wavelet analysis, we can define the approximated output as

        $$
            \begin{equation}
            f(\mathbf{x}) \approx \sum_{s, t} \left \langle f(\mathbf{x}), \phi_{s, t} (\mathbf{x} | a, b) \right \rangle \cdot \phi_{s, t} (\mathbf{x} | a, b),
            \end{equation}
        $$

        where $\phi_{s, t} (\cdot | a, b)$ denotes the child wavelet defined by hyper-parameters $a > 1$ and $b > 0$:

        $$
            \begin{equation}
            \phi_{s, t}(x | a, b) = \frac{1}{\sqrt{a^s}} \phi \left( \frac{x - t \cdot b \cdot a^s}{a^s} \right).
            \end{equation}
        $$

        Based on the wavelet mapping $\phi_{s, t} (\cdot | a, b)$, we can introduce the $1_{st}$-order and $2_{nd}$-order wavelet data expansion functions as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=1) = \left[ \phi_{0, 0}(\mathbf{x}), \phi_{0, 1}(\mathbf{x}), \cdots, \phi_{s, t}(\mathbf{x}) \right] \in R^{D_1}.
            \end{equation}
        $$

        and

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=2) = \kappa(\mathbf{x} | d=1) \otimes \kappa(\mathbf{x} | d=1) \in R^{D_2}.
            \end{equation}
        $$

        The output dimensions of the order-1 and order-2 wavelet expansions are $D_1 = s \cdot t \cdot m$ and $D_2 = (s \cdot t \cdot m)^2$, respectively.

        Specifically, the functions $\left\{ \phi_{s, t}\right\}_{ s, t \in Z}$ defines the orthonormal basis of the space and
        the mapping $\phi(\cdot)$ used in the child wavelet may have different representations.

        For Ricker wavelet, it can be represented as follows:

        __Ricker Wavelet:__

        $$
            \begin{equation}
            \begin{aligned}
            &\phi(\tau) = \frac{2 \left( 1 - \left( \frac{\tau}{\sigma} \right)^2 \right)}{\sqrt{3 \sigma} \pi^{\frac{1}{4}}} \exp\left(- \frac{\tau^2}{2 \sigma^2} \right).
            \end{aligned}
            \end{equation}
        $$

        Attributes
        ----------
        wavelet : callable
            Ricker wavelet function used during the transformation.

        Methods
        -------
        Inherits all methods from `discrete_wavelet_expansion`.
    """
    def __init__(self, name: str = 'ricker_wavelet_expansion', a: float = 1.0, b: float = 1.0, sigma: float = 1.0, *args, **kwargs):
        """
            Initializes the Ricker (Mexican hat) wavelet expansion.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'ricker_wavelet_expansion'.
            a : float, optional
                The scaling factor for the wavelet. Defaults to 1.0.
            b : float, optional
                The translation factor for the wavelet. Defaults to 1.0.
            sigma : float, optional
                Standard deviation for the wavelet. Defaults to 1.0.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = ricker_wavelet(a=a, b=b, sigma=sigma)


class dog_wavelet_expansion(discrete_wavelet_expansion):
    r"""
        Difference of Gaussians (DoG) Wavelet Expansion.

        Applies the Difference of Gaussians (DoG) wavelet function for feature expansion.

        Notes
        ----------
        Formally, given the input variable $\mathbf{x} \in R^{m}$, to approximate the underlying mapping $f: R^m \to R^n$ with wavelet analysis, we can define the approximated output as

        $$
            \begin{equation}
            f(\mathbf{x}) \approx \sum_{s, t} \left \langle f(\mathbf{x}), \phi_{s, t} (\mathbf{x} | a, b) \right \rangle \cdot \phi_{s, t} (\mathbf{x} | a, b),
            \end{equation}
        $$

        where $\phi_{s, t} (\cdot | a, b)$ denotes the child wavelet defined by hyper-parameters $a > 1$ and $b > 0$:

        $$
            \begin{equation}
            \phi_{s, t}(x | a, b) = \frac{1}{\sqrt{a^s}} \phi \left( \frac{x - t \cdot b \cdot a^s}{a^s} \right).
            \end{equation}
        $$

        Based on the wavelet mapping $\phi_{s, t} (\cdot | a, b)$, we can introduce the $1_{st}$-order and $2_{nd}$-order wavelet data expansion functions as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=1) = \left[ \phi_{0, 0}(\mathbf{x}), \phi_{0, 1}(\mathbf{x}), \cdots, \phi_{s, t}(\mathbf{x}) \right] \in R^{D_1}.
            \end{equation}
        $$

        and

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=2) = \kappa(\mathbf{x} | d=1) \otimes \kappa(\mathbf{x} | d=1) \in R^{D_2}.
            \end{equation}
        $$

        The output dimensions of the order-1 and order-2 wavelet expansions are $D_1 = s \cdot t \cdot m$ and $D_2 = (s \cdot t \cdot m)^2$, respectively.

        Specifically, the functions $\left\{ \phi_{s, t}\right\}_{ s, t \in Z}$ defines the orthonormal basis of the space and
        the mapping $\phi(\cdot)$ used in the child wavelet may have different representations.

        For DoG wavelet, it can be represented as follows:

        __DoG Wavelet:__

        $$
            \begin{equation}
            \begin{aligned}
            &\phi(\tau | \sigma_1, \sigma_2) = {P}(\tau | 0, \sigma_1) - {P}(\tau | 0, \sigma_2),
            \end{aligned}
            \end{equation}
        $$
        where ${P}(\cdot | 0, \sigma_1)$ denotes the PDF of the Gaussian distribution.

        Attributes
        ----------
        wavelet : callable
            DoG wavelet function used during the transformation.

        Methods
        -------
        Inherits all methods from `discrete_wavelet_expansion`.
    """
    def __init__(self, name: str = 'dog_wavelet_expansion', a: float = 1.0, b: float = 1.0, sigma_1: float = 1.0, sigma_2: float = 2.0, *args, **kwargs):
        """
            Initializes the Difference of Gaussians (DoG) wavelet expansion.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'dog_wavelet_expansion'.
            a : float, optional
                The scaling factor for the wavelet. Defaults to 1.0.
            b : float, optional
                The translation factor for the wavelet. Defaults to 1.0.
            sigma_1 : float, optional
                Standard deviation of the first Gaussian. Defaults to 1.0.
            sigma_2 : float, optional
                Standard deviation of the second Gaussian. Defaults to 2.0.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = dog_wavelet(a=a, b=b, sigma_1=sigma_1, sigma_2=sigma_2)


class meyer_wavelet_expansion(discrete_wavelet_expansion):
    r"""
        Meyer Wavelet Expansion.

        Applies the Meyer wavelet function for feature expansion.

        Notes
        ----------
        Formally, given the input variable $\mathbf{x} \in R^{m}$, to approximate the underlying mapping $f: R^m \to R^n$ with wavelet analysis, we can define the approximated output as

        $$
            \begin{equation}
            f(\mathbf{x}) \approx \sum_{s, t} \left \langle f(\mathbf{x}), \phi_{s, t} (\mathbf{x} | a, b) \right \rangle \cdot \phi_{s, t} (\mathbf{x} | a, b),
            \end{equation}
        $$

        where $\phi_{s, t} (\cdot | a, b)$ denotes the child wavelet defined by hyper-parameters $a > 1$ and $b > 0$:

        $$
            \begin{equation}
            \phi_{s, t}(x | a, b) = \frac{1}{\sqrt{a^s}} \phi \left( \frac{x - t \cdot b \cdot a^s}{a^s} \right).
            \end{equation}
        $$

        Based on the wavelet mapping $\phi_{s, t} (\cdot | a, b)$, we can introduce the $1_{st}$-order and $2_{nd}$-order wavelet data expansion functions as follows:

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=1) = \left[ \phi_{0, 0}(\mathbf{x}), \phi_{0, 1}(\mathbf{x}), \cdots, \phi_{s, t}(\mathbf{x}) \right] \in R^{D_1}.
            \end{equation}
        $$

        and

        $$
            \begin{equation}
            \kappa(\mathbf{x} | d=2) = \kappa(\mathbf{x} | d=1) \otimes \kappa(\mathbf{x} | d=1) \in R^{D_2}.
            \end{equation}
        $$

        The output dimensions of the order-1 and order-2 wavelet expansions are $D_1 = s \cdot t \cdot m$ and $D_2 = (s \cdot t \cdot m)^2$, respectively.

        Specifically, the functions $\left\{ \phi_{s, t}\right\}_{ s, t \in Z}$ defines the orthonormal basis of the space and
        the mapping $\phi(\cdot)$ used in the child wavelet may have different representations.

        For Meyer wavelet, it can be represented as follows:

        __Meyer Wavelet:__

        $$
            \begin{equation}
            \begin{aligned}
            &\phi(\tau) =
            \begin{cases}
            \frac{2}{3} + \frac{4}{3\pi} & \tau = 0,\\
            \frac{ \sin(\frac{2 \pi}{3} \tau) + \frac{4}{3} \tau \cos( \frac{4 \pi}{3} \tau) }{ \pi \tau - \frac{16 \pi}{9} \tau^3 } & \text{otherwise}.
            \end{cases}
            \end{aligned}
            \end{equation}
        $$

        Attributes
        ----------
        wavelet : callable
            Meyer wavelet function used during the transformation.

        Methods
        -------
        Inherits all methods from `discrete_wavelet_expansion`.
    """
    def __init__(self, name: str = 'meyer_wavelet_expansion', a: float = 1.0, b: float = 1.0, *args, **kwargs):
        """
            Initializes the Meyer wavelet expansion.

            Parameters
            ----------
            name : str, optional
                Name of the transformation. Defaults to 'meyer_wavelet_expansion'.
            a : float, optional
                The scaling factor for the wavelet. Defaults to 1.0.
            b : float, optional
                The translation factor for the wavelet. Defaults to 1.0.
            *args : tuple
                Additional positional arguments.
            **kwargs : dict
                Additional keyword arguments.
        """
        super().__init__(name=name, *args, **kwargs)
        self.wavelet = meyer_wavelet(a=a, b=b)
