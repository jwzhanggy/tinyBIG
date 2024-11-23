# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################
# Base Topology #
#################

import warnings
import torch
import pickle
import numpy as np
import scipy.sparse as sp

from tinybig.koala.linear_algebra import degree_based_normalize_matrix, sparse_mx_to_torch_sparse_tensor
from tinybig.util import create_directory_if_not_exists


class base_topology:
    """
    Base class for representing a topological structure such as a graph.

    Attributes
    ----------
    name : str
        Name of the topological structure.
    directed : bool
        Indicates if the graph is directed.
    nodes : dict
        Dictionary of nodes where keys are node identifiers, and values are indices.
    out_neighbors : dict
        Outgoing neighbors for each node.
    in_neighbors : dict
        Incoming neighbors for each node.
    node_attributes : dict, optional
        Attributes associated with each node.
    node_labels : dict, optional
        Labels associated with each node.
    link_attributes : dict, optional
        Attributes associated with each link.
    link_labels : dict, optional
        Labels associated with each link.
    device : str
        Device on which the structure is represented (e.g., 'cpu').

    Methods
    -------
    save(complete_path, cache_dir, output_file, *args, **kwargs)
        Saves the topology structure to a file.
    load(complete_path, cache_dir, output_file, *args, **kwargs)
        Loads a topology structure from a file.
    links_to_neighbors(links, node_dict)
        Converts a list of links to neighbor dictionaries.
    is_directed()
        Checks if the graph is directed.
    get_name()
        Returns the name of the topology.
    get_node_num()
        Returns the number of nodes in the topology.
    get_link_num()
        Returns the number of links in the topology.
    order()
        Alias for `get_node_num`.
    size()
        Alias for `get_link_num`.
    get_nodes()
        Returns the nodes in the topology.
    get_links()
        Returns the links in the topology.
    get_out_neighbors(node)
        Returns outgoing neighbors for a node.
    get_in_neighbors(node)
        Returns incoming neighbors for a node.
    get_neighbors(node)
        Returns all neighbors (both in and out) for a node.
    add_node(node)
        Adds a node to the topology.
    add_nodes(node_list)
        Adds multiple nodes to the topology.
    add_link(link)
        Adds a link to the topology.
    add_links(link_list)
        Adds multiple links to the topology.
    delete_node(node)
        Deletes a node and its associated links.
    delete_nodes(node_list)
        Deletes multiple nodes.
    delete_link(link)
        Deletes a specific link.
    delete_links(link_list)
        Deletes multiple links.
    get_node_attribute(node)
        Retrieves attributes for a specific node.
    get_node_label(node)
        Retrieves the label for a specific node.
    get_link_attribute(link)
        Retrieves attributes for a specific link.
    get_link_label(link)
        Retrieves the label for a specific link.
    to_matrix(self_dependence, self_scaling, normalization, normalization_mode, device, *args, **kwargs)
        Converts the topology into an adjacency matrix.

    """
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
        """
        Initializes the base topology structure.

        Parameters
        ----------
        name : str, optional
            The name of the topology, by default 'base_topological_structure'.
        nodes : list, optional
            A list of nodes to initialize the topology with. Each node is expected to be unique.
            If None, the topology starts with no nodes, by default None.
        links : list, optional
            A list of links where each link is represented as a tuple (n1, n2), indicating a directed edge from node n1 to node n2.
            If None, the topology starts with no links, by default None.
        directed : bool, optional
            Specifies whether the topology is directed. If True, links are treated as directed edges.
            Otherwise, links are bidirectional, by default True.
        node_attributes : dict, optional
            A dictionary where keys are nodes and values are their attributes. By default, None.
        node_labels : dict, optional
            A dictionary where keys are nodes and values are their labels. By default, None.
        link_attributes : dict, optional
            A dictionary where keys are links (tuples) and values are their attributes. By default, None.
        link_labels : dict, optional
            A dictionary where keys are links (tuples) and values are their labels. By default, None.
        device : str, optional
            The device on which the topology's data is stored, e.g., 'cpu', 'cuda', by default 'cpu'.

        Other Parameters
        ----------------
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Raises
        ------
        TypeError
            If `nodes` is not a list or None.
        """
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
        """
        Saves the topology structure to a file.

        Parameters
        ----------
        complete_path : str, optional
            Full path to save the file. If None, defaults to `cache_dir/output_file`.
        cache_dir : str
            Directory to save the file.
        output_file : str
            Name of the output file.

        Returns
        -------
        str
            Path to the saved file.
        """
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
        """
        Loads a topology structure from a file.

        Parameters
        ----------
        complete_path : str, optional
            Full path to load the file from. If None, defaults to `cache_dir/output_file`.
        cache_dir : str
            Directory to load the file from.
        output_file : str
            Name of the file to load.

        Returns
        -------
        base_topology
            The loaded topology structure.
        """
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
        """
        Converts a list of links into neighbor dictionaries.

        Parameters
        ----------
        links : list
            List of links, where each link is a tuple of (source, target).
        node_dict : dict
            Dictionary of nodes with their identifiers as keys.

        Returns
        -------
        tuple of dict
            Dictionaries of outgoing and incoming neighbors.
        """
        if links is None or node_dict is None:
            return {}, {}

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

    def is_directed(self):
        """
        Checks if the topology is directed.

        Returns
        -------
        bool
            True if the topology is directed, False otherwise.
        """
        return self.directed

    def get_name(self) -> str:
        """
        Returns the name of the topology.

        Returns
        -------
        str
            The name of the topology.
        """
        return self.name

    def get_node_num(self):
        """
        Returns the number of nodes in the topology.

        Returns
        -------
        int
            The number of nodes.
        """
        return self.order()

    def get_link_num(self):
        """
        Returns the number of links in the topology.

        Returns
        -------
        int
            The number of links.
        """
        return self.size()

    def order(self):
        """
        Alias for `get_node_num`.

        Returns
        -------
        int
            The number of nodes.
        """
        return len(self.nodes)

    def size(self):
        """
        Alias for `get_link_num`.

        Returns
        -------
        int
            The number of links.
        """
        return sum([len(self.out_neighbors[n]) for n in self.out_neighbors])

    def get_nodes(self):
        """
        Returns the nodes in the topology.

        Returns
        -------
        dict
            Dictionary of nodes.
        """
        return self.nodes

    def get_links(self):
        """
        Returns the links in the topology.

        Returns
        -------
        list of tuple
            List of links as (source, target) pairs.
        """
        links = [(n1, n2) for n1, n2_dict in self.out_neighbors.items() for n2 in n2_dict]
        if not self.directed:
            reverse_links = [(pair[1], pair[0]) for pair in links]
            links = list(set(links + reverse_links))
        return links

    def get_out_neighbors(self, node):
        """
        Returns the outgoing neighbors for a node.

        Parameters
        ----------
        node : any
            The node for which to get outgoing neighbors.

        Returns
        -------
        list
            List of outgoing neighbors.
        """
        if node in self.out_neighbors:
            return self.out_neighbors[node].keys()
        else:
            return []

    def get_in_neighbors(self, node):
        """
        Returns the incoming neighbors for a node.

        Parameters
        ----------
        node : any
            The node for which to get incoming neighbors.

        Returns
        -------
        list
            List of incoming neighbors.
        """
        if node in self.in_neighbors:
            return self.in_neighbors[node].keys()
        else:
            return []

    def get_neighbors(self, node):
        """
        Returns all neighbors (both in and out) for a node.

        Parameters
        ----------
        node : any
            The node for which to get neighbors.

        Returns
        -------
        list
            List of neighbors.
        """
        out_neighbors = self.get_out_neighbors(node)
        in_neighbors = self.get_in_neighbors(node)
        list(set(out_neighbors + in_neighbors))

    def add_node(self, node):
        """
        Adds a single node to the topology.

        Parameters
        ----------
        node : any
            The node to be added. If the node already exists, it is not added again.
        """
        if node not in self.nodes:
            self.nodes[node] = 1

    def add_nodes(self, node_list: dict | list | tuple):
        """
        Adds multiple nodes to the topology.

        Parameters
        ----------
        node_list : dict, list, or tuple
            Collection of nodes to add to the topology.
        """
        for node in node_list:
            self.add_node(node)

    def add_link(self, link: tuple):
        """
        Adds a single link to the topology.

        Parameters
        ----------
        link : tuple
            A tuple (n1, n2) representing a directed link from node n1 to node n2.
        """
        n1, n2 = link

        if n1 not in self.out_neighbors: self.out_neighbors[n1] = {}
        self.out_neighbors[n1][n2] = 1

        if n2 not in self.in_neighbors: self.in_neighbors[n2] = {}
        self.in_neighbors[n2][n1] = 1

        self.add_nodes([n1, n2])

    def add_links(self, link_list):
        """
        Adds multiple links to the topology.

        Parameters
        ----------
        link_list : list
            A list of tuples where each tuple represents a directed link (n1, n2).
        """
        for link in link_list:
            self.add_link(link)

    def delete_node(self, node):
        """
        Deletes a node and all associated links (incoming and outgoing).

        Parameters
        ----------
        node : any
            The node to delete.
        """
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
        """
        Deletes multiple nodes and all their associated links.

        Parameters
        ----------
        node_list : list
            A list of nodes to delete.
        """
        for node in node_list:
            self.delete_node(node)

    def delete_link(self, link):
        """
        Deletes a single link from the topology.

        Parameters
        ----------
        link : tuple
            A tuple (n1, n2) representing the link to delete.
        """
        n1, n2 = link
        if n1 in self.out_neighbors and n2 in self.out_neighbors[n1]:
            del self.out_neighbors[n1][n2]
        if n2 in self.in_neighbors and n1 in self.in_neighbors[n2]:
            del self.in_neighbors[n2][n1]

    def delete_links(self, link_list):
        """
        Deletes multiple links from the topology.

        Parameters
        ----------
        link_list : list
            A list of tuples where each tuple represents a directed link (n1, n2).
        """
        for link in link_list:
            self.delete_link(link)

    def get_node_attribute(self, node):
        """
        Retrieves attributes for a specific node.

        Parameters
        ----------
        node : any
            The node whose attributes are to be retrieved.

        Returns
        -------
        dict or None
            The attributes of the node. If the node or its attributes do not exist, returns None.
        """
        if node in self.nodes and node in self.node_attributes:
            return self.node_attributes[node]
        else:
            warnings.warn("The node doesn't exist in the node list or node attribute dictionary...")
            return None

    def get_node_label(self, node):
        """
        Retrieves the label for a specific node.

        Parameters
        ----------
        node : any
            The node whose label is to be retrieved.

        Returns
        -------
        any or None
            The label of the node. If the node or its label does not exist, returns None.
        """
        if node in self.nodes and node in self.node_labels:
            return self.node_labels[node]
        else:
            warnings.warn("The node doesn't exist in the node list or node label dictionary...")
            return None

    def get_link_attribute(self, link):
        """
        Retrieves attributes for a specific link.

        Parameters
        ----------
        link : tuple
            A tuple (n1, n2) representing the link whose attributes are to be retrieved.

        Returns
        -------
        dict or None
            The attributes of the link. If the link or its attributes do not exist, returns None.
        """
        n1, n2 = link
        if n1 in self.out_neighbors and n2 in self.out_neighbors[n1] and (n1, n2) in self.link_attributes:
            return self.link_attributes[(n1, n2)]
        else:
            warnings.warn("The link doesn't exist in the link list or link attribute dictionary...")
            return None

    def get_link_label(self, link):
        """
        Retrieves the label for a specific link.

        Parameters
        ----------
        link : tuple
            A tuple (n1, n2) representing the link whose label is to be retrieved.

        Returns
        -------
        any or None
            The label of the link. If the link or its label does not exist, returns None.
        """
        n1, n2 = link
        if n1 in self.out_neighbors and n2 in self.out_neighbors[n1] and (n1, n2) in self.link_labels:
            return self.link_labels[(n1, n2)]
        else:
            warnings.warn("The link doesn't exist in the link list or link label dictionary...")
            return None

    def to_matrix(self, self_dependence: bool = False, self_scaling: float = 1.0, normalization: bool = False, normalization_mode: str = 'row_column', device: str = 'cpu', *args, **kwargs):
        """
        Converts the topology into an adjacency matrix representation.

        Parameters
        ----------
        self_dependence : bool
            Whether to include self-loops in the adjacency matrix.
        self_scaling : float
            Scaling factor for self-loops, applicable if `self_dependence` is True.
        normalization : bool
            Whether to normalize the adjacency matrix.
        normalization_mode : str
            The mode of normalization. Possible values are 'row', 'column', or 'row_column'.
        device : str
            The device on which to create the adjacency matrix (e.g., 'cpu', 'cuda').

        Returns
        -------
        torch.Tensor or scipy.sparse.coo_matrix
            The adjacency matrix of the topology.
        dict
            A mapping between node IDs and their indices in the matrix.
        """
        node_id_index_map = self.nodes
        node_index_id_map = {index: node for node, index in node_id_index_map.items()}

        links = self.get_links()

        if device != 'mps':
            links = np.array(list(map(lambda pair: (node_id_index_map[pair[0]], node_id_index_map[pair[1]]), links)), dtype=np.int32)
            mx = sp.coo_matrix((np.ones(links.shape[0]), (links[:, 0], links[:, 1])), shape=(len(node_id_index_map), len(node_id_index_map)), dtype=np.float32)
            if not self.directed:
                mx = mx + mx.T.multiply(mx.T > mx) - mx.multiply(mx.T > mx)
            if self_dependence:
                mx += self_scaling * sp.eye(mx.shape[0])
            mx = sparse_mx_to_torch_sparse_tensor(mx)
        else:
            links = torch.tensor(list(map(lambda pair: (node_id_index_map[pair[0]], node_id_index_map[pair[1]]), links)), device=device)
            mx = torch.zeros((len(node_id_index_map), len(node_id_index_map)), device=device)
            mx[links[:, 0], links[:, 1]] = torch.ones(links.size(0), device=device)
            if not self.directed:
                mx = mx + mx.T * (mx.T > mx).float() - mx * (mx.T > mx).float()
            if self_dependence:
                mx += self_scaling * torch.eye(mx.shape[0], device=device)

        if normalization:
            mx = degree_based_normalize_matrix(mx=mx, mode=normalization_mode)

        return mx, {'node_id_index_map': node_id_index_map, 'node_index_id_map': node_index_id_map}

