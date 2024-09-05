# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
from typing import Tuple, Dict, Any

###############################
# Chain Topological Structure #
###############################

import numpy as np
import scipy.sparse as sp

from tinybig.koala.topology import base_topology
from tinybig.koala.linear_algebra import normalize_matrix


class chain(base_topology):

    def __init__(self, links: list | dict = None, length: int = None, name: str = 'chain', bi_directional: bool = False, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        if links is not None:
            chain_links = links
        elif length is not None:
            chain_links = [(i, i + 1) for i in range(length)]
        else:
            raise ValueError('Either links or length must be specified')

        self.out_neighbors, self.in_neighbors = self.links_to_sequence(chain_links)
        self.bi_directional = bi_directional

    @staticmethod
    def links_to_sequence(links: dict | list):
        out_neighbors = {}
        in_neighbors = {}
        for (n1, n2) in links:
            if n1 not in out_neighbors: out_neighbors[n1] = {}
            out_neighbors[n1][n2] = 1
            if n2 not in in_neighbors: in_neighbors[n2] = {}
            in_neighbors[n2][n1] = 1
        return out_neighbors, in_neighbors

    def order(self):
        return len(self.out_neighbors) + 1

    def size(self):
        return len(self.out_neighbors)

    def get_links(self):
        links = [(n1, n2) for n1, n2_dict in self.out_neighbors.items() for n2 in n2_dict]
        if self.bi_directional:
            reverse_links = [(pair[1], pair[0]) for pair in links]
            links = list(set(links + reverse_links))
        return links

    def to_matrix(self, normalization: bool = False, mode: str = 'row-column', *args, **kwargs):
        node_id_index_map = {}
        node_index_id_map = {}
        index = 0
        for node in self.out_neighbors:
            node_id_index_map[node] = index
            node_index_id_map[index] = node
            index += 1

        links = self.get_links()

        links = np.array(list(map(lambda pair: (node_id_index_map[pair[0]], node_id_index_map[pair[1]]), links)))
        adj = sp.coo_matrix((np.ones(links.shape[0]), (links[:, 0], links[:, 1])), shape=(len(node_id_index_map), len(node_id_index_map)), dtype=np.float32)

        if normalization:
            adj = normalize_matrix(mx=adj, mode=mode)

        return adj

