# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################
# Base Topology #
#################

import warnings
import torch
import pickle

from tinybig.koala.linear_algebra import degree_based_normalize_matrix
from tinybig.util import create_directory_if_not_exists

class base_topology:

    def __init__(
        self,
        name: str = 'base_topological_structure',
        nodes: list = None,
        links: list = None,
        directed: bool = True,
        node_attributes: dict = None,
        node_labels: dict = None,
        link_attributes: dict = None,
        link_labels: dict = None,
        device: str = 'cpu',
        *args, **kwargs
    ):
        self.name = name
        self.directed = directed

        if nodes is None:
            self.nodes = {}
        elif isinstance(nodes, list):
            self.nodes = {node: index for index, node in enumerate(nodes)}
        else:
            raise TypeError('nodes must be a list...')

        self.out_neighbors, self.in_neighbors = self.links_to_neighbors(links, self.nodes)

        self.node_attributes = node_attributes
        self.node_labels = node_labels
        self.link_attributes = link_attributes
        self.link_labels = link_labels

        self.device = device

    def save(self, complete_path: str = None, cache_dir='./data', output_file='data_screenshot', *args, **kwargs):
        path = complete_path if complete_path is not None else f'{cache_dir}/{output_file}'
        create_directory_if_not_exists(path)
        data = {
            'name': self.name,
            'directed': self.directed,
            'nodes': self.nodes,
            'out_neighbors': self.out_neighbors,
            'in_neighbors': self.in_neighbors,
            'node_attributes': self.node_attributes,
            'node_labels': self.node_labels,
            'link_attributes': self.link_attributes,
            'link_labels': self.link_labels,
            'device': self.device,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return path

    @staticmethod
    def load(complete_path: str = None, cache_dir: str = './data', output_file: str = 'graph_screenshot_data', *args, **kwargs):
        path = complete_path if complete_path is not None else f'{cache_dir}/{output_file}'
        with open(path, 'rb') as f:
            data = pickle.load(f)

        topology_structure = base_topology()
        topology_structure.name = data['name']
        topology_structure.directed = data['directed']
        topology_structure.nodes = data['nodes']
        topology_structure.out_neighbors = data['out_neighbors']
        topology_structure.in_neighbors = data['in_neighbors']
        topology_structure.node_attributes = data['node_attributes']
        topology_structure.node_labels = data['node_labels']
        topology_structure.link_attributes = data['link_attributes']
        topology_structure.link_labels = data['link_labels']
        topology_structure.device = data['device']

        return topology_structure


    @staticmethod
    def links_to_neighbors(links: list, node_dict: dict):
        out_neighbors = {}
        in_neighbors = {}
        for (n1, n2) in links:
            if n1 in node_dict:
                if n1 not in out_neighbors:
                    out_neighbors[n1] = {}
                out_neighbors[n1][n2] = 1
            if n2 in node_dict:
                if n2 not in in_neighbors:
                    in_neighbors[n2] = {}
                in_neighbors[n2][n1] = 1
        return out_neighbors, in_neighbors

    def get_name(self) -> str:
        return self.name

    def get_node_num(self):
        return self.order()

    def get_link_num(self):
        return self.size()

    def order(self):
        return len(self.nodes)

    def size(self):
        return sum([len(self.out_neighbors[n]) for n in self.out_neighbors])

    def get_nodes(self):
        return self.nodes

    def get_links(self):
        links = [(n1, n2) for n1, n2_dict in self.out_neighbors.items() for n2 in n2_dict]
        if not self.directed:
            reverse_links = [(pair[1], pair[0]) for pair in links]
            links = list(set(links + reverse_links))
        return links

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

        self.add_nodes([n1, n2])

    def add_links(self, link_list):
        for link in link_list:
            self.add_link(link)

    def delete_node(self, node):
        if node in self.nodes:
            del self.nodes[node]

        node_out_neighbors = self.out_neighbors[node] if node in self.out_neighbors else {}
        node_in_neighbors = self.in_neighbors[node] if node in self.in_neighbors else {}

        if node in self.out_neighbors:
            del self.out_neighbors[node]
            for n in node_in_neighbors:
                del self.out_neighbors[n][node]
                if len(self.out_neighbors[n]) == 0:
                    del self.out_neighbors[n]

        if node in self.in_neighbors:
            del self.in_neighbors[node]
            for n in node_out_neighbors:
                del self.in_neighbors[n][node]
                if len(self.in_neighbors[n]) == 0:
                    del self.in_neighbors[n]

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

    def get_node_attribute(self, node):
        if node in self.nodes and node in self.node_attributes:
            return self.node_attributes[node]
        else:
            warnings.warn("The node doesn't exist in the node list or node attribute dictionary...")
            return None

    def get_node_label(self, node):
        if node in self.nodes and node in self.node_labels:
            return self.node_labels[node]
        else:
            warnings.warn("The node doesn't exist in the node list or node label dictionary...")
            return None

    def get_link_attribute(self, link):
        n1, n2 = link
        if n1 in self.out_neighbors and n2 in self.out_neighbors[n1] and (n1, n2) in self.link_attributes:
            return self.link_attributes[(n1, n2)]
        else:
            warnings.warn("The link doesn't exist in the link list or link attribute dictionary...")
            return None

    def get_link_label(self, link):
        n1, n2 = link
        if n1 in self.out_neighbors and n2 in self.out_neighbors[n1] and (n1, n2) in self.link_labels:
            return self.link_labels[(n1, n2)]
        else:
            warnings.warn("The link doesn't exist in the link list or link label dictionary...")
            return None

    def to_matrix(self, normalization: bool = False, normalization_mode: str = 'row_column', to_sparse: bool = False, device: str = 'cpu', *args, **kwargs):
        node_id_index_map = self.nodes
        node_index_id_map = {index: node for node, index in node_id_index_map.items()}

        links = self.get_links()

        links = torch.tensor(list(map(lambda pair: (node_id_index_map[pair[0]], node_id_index_map[pair[1]]), links)), device=device)

        if to_sparse and device != 'mps':
            mx = torch.sparse_coo_tensor(torch.tensor([links[:, 0], links[:, 1]]), values=torch.ones(links.shape[0]), size=(len(node_id_index_map), len(node_id_index_map)), device=device)
        else:
            mx = torch.zeros((len(node_id_index_map), len(node_id_index_map)), device=device)
            mx[links[:, 0], links[:, 1]] = torch.ones(links.size(0), device=device)

        if normalization:
            mx = degree_based_normalize_matrix(mx=mx, mode=normalization_mode)

        return mx, {'node_id_index_map': node_id_index_map, 'node_index_id_map': node_index_id_map}

