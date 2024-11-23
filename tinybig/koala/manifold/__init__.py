# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################
# Manifold Library #
####################

"""

This module provides the libraries of "manifold" that can be used to build the RPN model within the tinyBIG toolkit.

## Manifold Library

In mathematics, a manifold is a mesmerizing geometric structure that generalizes our intuitive understanding of curves
and surfaces to higher dimensions, offering a bridge between local simplicity and global complexity.

## Functions Implementation

Currently, the functions implemented in this library include

* manifold,
* isomap_manifold,
* tsne_manifold,
* spectral_embedding_manifold,
* mds_manifold,
* lle_manifold,

"""

from tinybig.koala.manifold.manifold import (
    manifold,
    isomap_manifold,
    tsne_manifold,
    spectral_embedding_manifold,
    mds_manifold,
    lle_manifold,
)