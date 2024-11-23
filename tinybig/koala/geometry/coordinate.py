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
    """
        Represents a coordinate in an N-dimensional space.

        Attributes
        ----------
        coords : tuple[int, ...]
            The tuple representing the coordinate in N-dimensional space.

        Methods
        -------
        __add__(other)
            Adds another coordinate to this one.
        __sub__(other)
            Subtracts another coordinate from this one.
        __repr__()
            Returns a string representation of the coordinate.
        __eq__(other)
            Checks equality of two coordinates.
        __lt__(other)
            Compares if this coordinate is less than another.
        __le__(other)
            Compares if this coordinate is less than or equal to another.
        __gt__(other)
            Compares if this coordinate is greater than another.
        __ge__(other)
            Compares if this coordinate is greater than or equal to another.
        __hash__()
            Returns the hash of the coordinate.
        dimension()
            Returns the dimension of the coordinate.
        distance_to(other, kernel_name='euclidean_distance')
            Calculates the distance to another coordinate using a specified kernel.
    """
    def __init__(self, coords: tuple[int, ...], *args, **kwargs):
        """
            Initializes a generic N-dimensional coordinate object.

            This class serves as a base for representing a point in an N-dimensional
            space, defined by a tuple of integer coordinates.

            Parameters
            ----------
            coords : tuple[int, ...]
                A tuple representing the N-dimensional coordinates of the point.
            *args, **kwargs
                Additional arguments for future extensibility.

            Attributes
            ----------
            coords : tuple[int, ...]
                Stores the coordinates of the point in N-dimensional space.

        """
        self.coords = coords

    def __add__(self, other):
        """
            Adds another coordinate to this one.

            Parameters
            ----------
            other : coordinate
                The coordinate to add.

            Returns
            -------
            coordinate
                The resulting coordinate after addition.

            Raises
            ------
            TypeError
                If the operands are not of the same type or dimensions.
        """
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(tuple(a + b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    # Redefine the - operator
    def __sub__(self, other):
        """
            Subtracts another coordinate from this one.

            Parameters
            ----------
            other : coordinate
                The coordinate to subtract.

            Returns
            -------
            coordinate
                The resulting coordinate after subtraction.

            Raises
            ------
            TypeError
                If the operands are not of the same type or dimensions.
        """
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(tuple(a - b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    def __repr__(self):
        """
            Returns a string representation of the coordinate.

            Returns
            -------
            str
                The string representation of the coordinate.
        """
        return f"{self.__class__.__name__}{self.coords}"

    # Optionally, add equality checking
    def __eq__(self, other):
        """
            Checks equality of two coordinates.

            Parameters
            ----------
            other : coordinate
                The coordinate to compare.

            Returns
            -------
            bool
                True if coordinates are equal, False otherwise.
        """
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords == other.coords

    def __lt__(self, other):
        """
            Checks if this coordinate is less than another coordinate.

            Parameters
            ----------
            other : coordinate
                The coordinate to compare.

            Returns
            -------
            bool
                True if this coordinate is less than or equal to the other coordinate, False otherwise.

            Raises
            ------
            TypeError
                If the coordinates are not of the same type or dimension.
        """
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords < other.coords

    def __le__(self, other):
        """
            Checks if this coordinate is less than or equal to another coordinate.

            Parameters
            ----------
            other : coordinate
                The coordinate to compare.

            Returns
            -------
            bool
                True if this coordinate is less than or equal to the other coordinate, False otherwise.

            Raises
            ------
            TypeError
                If the coordinates are not of the same type or dimension.
        """
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords <= other.coords

    def __gt__(self, other):
        """
            Checks if this coordinate is greater than another coordinate.

            Parameters
            ----------
            other : coordinate
                The coordinate to compare.

            Returns
            -------
            bool
                True if this coordinate is greater than the other coordinate, False otherwise.

            Raises
            ------
            TypeError
                If the coordinates are not of the same type or dimension.
        """
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords > other.coords

    def __ge__(self, other):
        """
            Checks if this coordinate is greater than or equal to another coordinate.

            Parameters
            ----------
            other : coordinate
                The coordinate to compare.

            Returns
            -------
            bool
                True if this coordinate is greater than or equal to the other coordinate, False otherwise.

            Raises
            ------
            TypeError
                If the coordinates are not of the same type or dimension.
        """
        return isinstance(other, self.__class__) and self.dimension() == other.dimension() and self.coords >= other.coords

    def __hash__(self):
        """
            Returns the hash value of the coordinate.

            Returns
            -------
            int
                The hash value based on the coordinate's tuple representation.
        """
        return hash(self.coords)

    def dimension(self):
        """
            Returns the dimension of the coordinate.

            Returns
            -------
            int
                The dimension of the coordinate.
        """
        return len(self.coords)

    def distance_to(self, other, kernel_name: str = 'euclidean_distance'):
        """
            Calculates the distance to another coordinate using a specified kernel.

            Parameters
            ----------
            other : coordinate
                The coordinate to calculate the distance to.
            kernel_name : str, optional
                The name of the kernel to use for distance calculation (default is 'euclidean_distance').

            Returns
            -------
            float
                The distance to the other coordinate.

            Raises
            ------
            TypeError
                If the operands are not of the same type or dimensions.
        """
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return kernel(kernel_name=kernel_name, x=torch.tensor(self.coords, dtype=torch.float), x2=torch.tensor(other.coords, dtype=torch.float)).item()
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")


class coordinate_3d(coordinate):
    """
        Represents a coordinate in a 3D space, extending the general coordinate class.

        Attributes
        ----------
        coords : tuple[int, int, int]
            The tuple representing the 3D coordinate.
        h : int
            The height coordinate.
        w : int
            The width coordinate.
        d : int
            The depth coordinate.
        x, y, z : int
            Aliases for h, w, and d respectively.

        Methods
        -------
        __add__(other)
            Adds another 3D coordinate to this one.
        __sub__(other)
            Subtracts another 3D coordinate from this one.
    """
    def __init__(self, h: int, w: int, d: int, *args, **kwargs):
        """
            Initializes a 3D coordinate object.

            Parameters
            ----------
            h : int
                The height coordinate.
            w : int
                The width coordinate.
            d : int
                The depth coordinate.
            *args, **kwargs
                Additional arguments for customization.

            Attributes
            ----------
            coords : tuple[int, int, int]
                Stores the 3D coordinate as a tuple.
        """
        super().__init__(coords=(h, w, d), *args, **kwargs)

    def __add__(self, other):
        """
            Adds another 3D coordinate to this one.

            Parameters
            ----------
            other : coordinate_3d
                The 3D coordinate to add.

            Returns
            -------
            coordinate_3d
                The resulting coordinate after addition.

            Raises
            ------
            TypeError
                If the operands are not of the same type or dimensions.
        """
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(*tuple(a + b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    # Redefine the - operator
    def __sub__(self, other):
        """
            Subtracts another 3D coordinate from this one.

            Parameters
            ----------
            other : coordinate_3d
                The 3D coordinate to subtract.

            Returns
            -------
            coordinate_3d
                The resulting coordinate after subtraction.

            Raises
            ------
            TypeError
                If the operands are not of the same type or dimensions.
        """
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(*tuple(a - b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    @property
    def h(self):
        """
            The height coordinate.

            Returns
            -------
            int
                The height value.
        """
        return self.coords[0] if self.dimension() > 0 else None

    @h.setter
    def h(self, value):
        """
            Sets the height coordinate.

            Parameters
            ----------
            value : int
                The new value for the height coordinate.
        """
        if self.dimension() > 0:
            self.coords = (value,) + self.coords[1:]

    @property
    def w(self):
        """
            The width coordinate.

            Returns
            -------
            int
                The width value.
        """
        return self.coords[1] if self.dimension() > 1 else None

    @w.setter
    def w(self, value):
        """
            Sets the width coordinate.

            Parameters
            ----------
            value : int
                The new value for the width coordinate.
        """
        if self.dimension() > 1:
            self.coords = self.coords[0:1] + (value,) + self.coords[2:]

    @property
    def d(self):
        """
            The depth coordinate.

            Returns
            -------
            int
                The depth value.
        """
        return self.coords[2] if self.dimension() > 2 else None

    @d.setter
    def d(self, value):
        """
            Sets the depth coordinate.

            Parameters
            ----------
            value : int
                The new value for the depth coordinate.
        """
        if self.dimension() > 2:
            self.coords = self.coords[0:2] + (value,)

    # Aliases for x, y, z: h, w, d
    @property
    def x(self):
        """
            Alias for h (height).
        """
        return self.h

    @x.setter
    def x(self, value):
        """
            Alias for setting the height coordinate.

            Parameters
            ----------
            value : int
                The new value for the height coordinate.
        """
        self.h = value

    @property
    def y(self):
        """
            Alias for w (width).
        """
        return self.w

    @y.setter
    def y(self, value):
        """
            Alias for setting the width coordinate.

            Parameters
            ----------
            value : int
                The new value for the width coordinate.
        """
        self.w = value

    @property
    def z(self):
        """
            Alias for d (depth).
        """
        return self.d

    @z.setter
    def z(self, value):
        """
            Alias for setting the depth coordinate.

            Parameters
            ----------
            value : int
                The new value for the depth coordinate.
        """
        self.d = value


class coordinate_2d(coordinate):
    """
        Represents a coordinate in a 2D space, extending the general coordinate class.

        Attributes
        ----------
        coords : tuple[int, int]
            The tuple representing the 2D coordinate.
        h : int
            The height coordinate.
        w : int
            The width coordinate.
        x, y : int
            Aliases for h and w respectively.

        Methods
        -------
        __add__(other)
            Adds another 2D coordinate to this one.
        __sub__(other)
            Subtracts another 2D coordinate from this one.
    """
    def __init__(self, h: int, w: int, *args, **kwargs):
        """
            Initializes a 2D coordinate object.

            Parameters
            ----------
            h : int
                The height coordinate.
            w : int
                The width coordinate.
            *args, **kwargs
                Additional arguments for customization.

            Attributes
            ----------
            coords : tuple[int, int]
                Stores the 2D coordinate as a tuple.
        """
        super().__init__(coords=(h, w), *args, **kwargs)

    def __add__(self, other):
        """
            Adds another 2D coordinate to this one.

            Parameters
            ----------
            other : coordinate_2d
                The 2D coordinate to add.

            Returns
            -------
            coordinate_2d
                The resulting coordinate after addition.

            Raises
            ------
            TypeError
                If the operands are not of the same type or dimensions.
        """
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(*tuple(a + b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    # Redefine the - operator
    def __sub__(self, other):
        """
            Subtracts another 2D coordinate from this one.

            Parameters
            ----------
            other : coordinate_2d
                The 2D coordinate to subtract.

            Returns
            -------
            coordinate_2d
                The resulting coordinate after subtraction.

            Raises
            ------
            TypeError
                If the operands are not of the same type or dimensions.
        """
        if isinstance(other, self.__class__) and len(self.coords) == len(other.coords):
            return self.__class__(*tuple(a - b for a, b in zip(self.coords, other.coords)))
        raise TypeError("Operands must be of type Coordinate and have the same dimensions")

    @property
    def h(self):
        """
            The height coordinate.

            Returns
            -------
            int
                The height value.
        """
        return self.coords[0] if self.dimension() > 0 else None

    @h.setter
    def h(self, value):
        """
            Sets the height coordinate.

            Parameters
            ----------
            value : int
                The new value for the height coordinate.
        """
        if self.dimension() > 0:
            self.coords = (value,) + self.coords[1:]

    @property
    def w(self):
        """
            The width coordinate.

            Returns
            -------
            int
                The width value.
        """
        return self.coords[1] if self.dimension() > 1 else None

    @w.setter
    def w(self, value):
        """
            Sets the width coordinate.

            Parameters
            ----------
            value : int
                The new value for the width coordinate.
        """
        if self.dimension() > 1:
            self.coords = self.coords[0:1] + (value,) + self.coords[2:]

    @property
    def x(self):
        """
            Alias for h (height).
        """
        return self.h

    @x.setter
    def x(self, value):
        """
            Alias for setting the height coordinate.

            Parameters
            ----------
            value : int
                The new value for the height coordinate.
        """
        self.h = value

    @property
    def y(self):
        """
            Alias for w (width).
        """
        return self.w

    @y.setter
    def y(self, value):
        """
            Alias for setting the width coordinate.

            Parameters
            ----------
            value : int
                The new value for the width coordinate.
        """
        self.w = value

