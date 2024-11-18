# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###########################
# Basic Algebra Functions #
###########################

import math


def find_close_factors(n: int):
    sqrt_n = int(math.isqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i
    return None
