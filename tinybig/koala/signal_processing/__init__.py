# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############################
# Signal Processing Library #
#############################

"""

This module provides the libraries of "signal processing" that can be used to build the RPN model within the tinyBIG toolkit.

## Signal Processing Library

Signal processing is a fascinating field that lies at the intersection of mathematics, electrical engineering,
and computer science, focusing on the analysis, manipulation, and synthesis of signals that carry information about our world.

## Functions Implementation

Currently, the functions implemented in this library include:

* discrete_wavelet,
* harr_wavelet,
* beta_wavelet,
* dog_wavelet,
* meyer_wavelet,
* ricker_wavelet,
* shannon_wavelet

"""

from tinybig.koala.signal_processing.wavelet import (
    discrete_wavelet,
    harr_wavelet,
    beta_wavelet,
    dog_wavelet,
    meyer_wavelet,
    ricker_wavelet,
    shannon_wavelet
)