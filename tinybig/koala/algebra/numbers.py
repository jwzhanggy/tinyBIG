# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################
# Base Geometry #
#################

import math


def find_close_factors(n: int):
    sqrt_n = int(math.isqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i
    return None


if __name__ == '__main__':
    n = 99
    factors = find_close_factors(n)
    if factors:
        print(f"{n} can be divided into factors: {factors}")
    else:
        print(f"No close factors found for {n}")