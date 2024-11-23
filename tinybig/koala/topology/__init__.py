# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################
# Topology Library #
####################

"""

This module provides the libraries of "topology" that can be used to build the RPN model within the tinyBIG toolkit.

## Topology Library

Topology is an enchanting branch of mathematics that studies the properties of geometric objects that remain unchanged
under continuous deformation, earning it the playful nickname of "rubber-sheet geometry."

## Functions Implementation

Currently, the functions implemented in this library include:

* base topology
* graph
* chain

"""

from tinybig.koala.topology.base_topology import base_topology
from tinybig.koala.topology.graph import graph
from tinybig.koala.topology.chain import chain
from tinybig.koala.geometry.grid import grid