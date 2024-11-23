# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################
# Geometry Library #
####################

"""
This module provides the libraries of "geometry" that can be used to build the RPN model within the tinyBIG toolkit.

## Geometry Library

Geometry is a branch of mathematics concerned with properties of space such as the distance, shape, size, and relative position of figures.
Geometry is, along with arithmetic, one of the oldest branches of mathematics.
A mathematician who works in the field of geometry is called a geometer.
Until the 19th century, geometry was almost exclusively devoted to Euclidean geometry, which includes the notions of
point, line, plane, distance, angle, surface, and curve, as fundamental concepts.

## Functions Implementation

Currently, the functions implemented in this library include

* coordinate, coordinate_3d, coordinate_2d
* geometric_space
* grid
* cuboid
* cylinder
* sphere

"""

from tinybig.koala.geometry.coordinate import coordinate, coordinate_3d, coordinate_2d
from tinybig.koala.geometry.base_geometry import geometric_space
from tinybig.koala.geometry.cuboid import cuboid
from tinybig.koala.geometry.cylinder import cylinder
from tinybig.koala.geometry.sphere import sphere
from tinybig.koala.geometry.grid import grid
