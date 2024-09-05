# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
from typing import Tuple, Dict, Any

###############################
# Graph Topological Structure #
###############################

import numpy as np
import scipy.sparse as sp

from tinybig.koala.topology import base_topology
from tinybig.koala.linear_algebra import normalize_matrix


class graph(base_topology):

    def __init__(self, nodes: dict | list, links: dict | list, node_attribute: dict = None, node_label: dict = None, link_attribute: dict = None, link_label: dict = None, directed: bool = True, name: str = 'graph', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.directed = directed

        if isinstance(nodes, list):
            self.nodes = {node: 1 for node in nodes}
        elif isinstance(nodes, dict):
            self.nodes = nodes

        self.out_neighbors, self.in_neighbors = self.links_to_neighbors(links)

        self.node_attribute = node_attribute
        self.node_label = node_label
        self.link_attribute = link_attribute
        self.link_label = link_label

    @staticmethod
    def links_to_neighbors(links: dict | list):
        out_neighbors = {}
        in_neighbors = {}
        for (n1, n2) in links:
            if n1 not in out_neighbors: out_neighbors[n1] = {}
            out_neighbors[n1][n2] = 1
            if n2 not in in_neighbors: in_neighbors[n2] = {}
            in_neighbors[n2][n1] = 1
        return out_neighbors, in_neighbors

    def get_out_neighbors(self, node):
        if node in self.out_neighbors:
            return self.out_neighbors[node].keys()
        else:
            return []

    def get_in_neighbors(self, node):
        if node in self.in_neighbors:
            return self.in_neighbors[node].keys()
        else:
            return []

    def get_neighbors(self, node):
        out_neighbors = self.get_out_neighbors(node)
        in_neighbors = self.get_in_neighbors(node)
        list(set(out_neighbors + in_neighbors))

    def get_nodes(self):
        return self.nodes

    def get_links(self):
        links = [(n1, n2) for n1, n2_dict in self.out_neighbors.items() for n2 in n2_dict]
        if not self.directed:
            reverse_links = [(pair[1], pair[0]) for pair in links]
            links = list(set(links + reverse_links))
        return links

    def order(self):
        return len(self.nodes)

    def size(self):
        return np.sum([len(self.out_neighbors[n]) for n in self.out_neighbors])

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node] = 1

    def add_nodes(self, node_list: dict | list | tuple):
        for node in node_list:
            self.add_node(node)

    def add_link(self, link: tuple):
        n1, n2 = link

        if n1 not in self.out_neighbors: self.out_neighbors[n1] = {}
        self.out_neighbors[n1][n2] = 1

        if n2 not in self.in_neighbors: self.in_neighbors[n2] = {}
        self.in_neighbors[n2][n1] = 1

        self.add_nodes(link)

    def add_links(self, link_list):
        for link in link_list:
            self.add_link(link)

    def delete_node(self, node):
        if node in self.nodes:
            del self.nodes[node]

        node_out_neighbors = self.out_neighbors[node]
        node_in_neighbors = self.in_neighbors[node]

        if node in self.out_neighbors:
            del self.out_neighbors[node]
            for n in node_in_neighbors:
                del self.out_neighbors[n][node]

        if node in self.in_neighbors:
            del self.in_neighbors[node]
            for n in node_out_neighbors:
                del self.in_neighbors[n][node]

    def delete_nodes(self, node_list):
        for node in node_list:
            self.delete_node(node)

    def delete_link(self, link):
        n1, n2 = link
        if n1 in self.out_neighbors and n2 in self.out_neighbors[n1]:
            del self.out_neighbors[n1][n2]
        if n2 in self.in_neighbors and n1 in self.in_neighbors[n2]:
            del self.in_neighbors[n2][n1]

    def delete_links(self, link_list):
        for link in link_list:
            self.delete_link(link)

    def to_matrix(self, normalization: bool = False, mode: str = 'row-column', *args, **kwargs):
        node_id_index_map = {i: j for i, j in enumerate(self.nodes.keys())}
        node_index_id_map = {j: i for i, j in enumerate(self.nodes.keys())}

        links = self.get_links()

        links = np.array(list(map(lambda pair: (node_id_index_map[pair[0]], node_id_index_map[pair[1]]), links)))
        adj = sp.coo_matrix((np.ones(links.shape[0]), (links[:, 0], links[:, 1])), shape=(len(node_id_index_map), len(node_id_index_map)), dtype=np.float32)

        if normalization:
            adj = normalize_matrix(mx=adj, mode=mode)

        return adj, {'node_id_index_map': node_id_index_map, 'node_index_id_map': node_index_id_map}

