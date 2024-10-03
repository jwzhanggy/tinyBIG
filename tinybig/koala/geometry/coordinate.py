# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

########################
# Geometric Coordinate #
########################

import numpy as np
import torch

from tinybig.koala.linear_algebra import kernel


class coordinate:
    def __init__(self, coords: tuple[int, ...], *args, **kwargs):
        self.coords = coords

    def __add__(self, other):
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(tuple(a + b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    # Redefine the - operator
    def __sub__(self, other):
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(tuple(a - b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    def __repr__(self):
        return f"{self.__class__.__name__}{self.coords}"

    # Optionally, add equality checking
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords == other.coords

    def __lt__(self, other):
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords < other.coords

    def __le__(self, other):
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords <= other.coords

    def __gt__(self, other):
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords > other.coords

    def __ge__(self, other):
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords >= other.coords

    def __hash__(self):
        return hash(self.coords)

    def dimension(self):
        return len(self.coords)

    def distance_to(self, other, kernel_name: str = 'euclidean_distance'):
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return kernel(kernel_name=kernel_name, x=torch.tensor(self.coords, dtype=torch.float), x2=torch.tensor(other.coords, dtype=torch.float)).item()
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")


class coordinate_3d(coordinate):
    def __init__(self, h: int, w: int, d: int, *args, **kwargs):
        super().__init__(coords=(h, w, d), *args, **kwargs)

    def __add__(self, other):
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(*tuple(a + b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    # Redefine the - operator
    def __sub__(self, other):
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(*tuple(a - b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    @property
    def h(self):
        return self.coords[0] if self.dimension() > 0 else None

    @h.setter
    def h(self, value):
        if self.dimension() > 0:
            self.coords = (value,) + self.coords[1:]

    @property
    def w(self):
        return self.coords[1] if self.dimension() > 1 else None

    @w.setter
    def w(self, value):
        if self.dimension() > 1:
            self.coords = self.coords[0:1] + (value,) + self.coords[2:]

    @property
    def d(self):
        return self.coords[2] if self.dimension() > 2 else None

    @d.setter
    def d(self, value):
        if self.dimension() > 2:
            self.coords = self.coords[0:2] + (value,)

    # Aliases for x, y, z: h, w, d
    @property
    def x(self):
        return self.h

    @x.setter
    def x(self, value):
        self.h = value

    @property
    def y(self):
        return self.w

    @y.setter
    def y(self, value):
        self.w = value

    @property
    def z(self):
        return self.d

    @z.setter
    def z(self, value):
        self.d = value


class coordinate_2d(coordinate):

    def __init__(self, h: int, w: int, *args, **kwargs):
        super().__init__(coords=(h, w), *args, **kwargs)

    def __add__(self, other):
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(*tuple(a + b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    # Redefine the - operator
    def __sub__(self, other):
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(*tuple(a - b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    @property
    def h(self):
        return self.coords[0] if self.dimension() > 0 else None

    @h.setter
    def h(self, value):
        if self.dimension() > 0:
            self.coords = (value,) + self.coords[1:]

    @property
    def w(self):
        return self.coords[1] if self.dimension() > 1 else None

    @w.setter
    def w(self, value):
        if self.dimension() > 1:
            self.coords = self.coords[0:1] + (value,) + self.coords[2:]

    @property
    def x(self):
        return self.h

    @x.setter
    def x(self, value):
        self.h = value

    @property
    def y(self):
        return self.w

    @y.setter
    def y(self, value):
        self.w = value

