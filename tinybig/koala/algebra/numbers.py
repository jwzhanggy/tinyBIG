# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################
# Basic Algebra Functions #
###########################

import math


def find_close_factors(n: int):
    """
        Find the largest factor of a given integer `n` that is closest to its square root.

        This function identifies the largest integer factor of `n` less than or equal
        to the square root of `n`.

        Parameters
        ----------
        n : int
            The integer whose factor is to be found.

        Returns
        -------
        int or None
            The largest factor of `n` closest to its square root, or `None` if no such factor exists.

        Raises
        ------
        ValueError
            If `n` is not a positive integer.

    """
    sqrt_n = int(math.isqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i
    return None
