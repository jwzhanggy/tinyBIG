# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################
# Graph Dataloader #
####################


import warnings
from abc import abstractmethod

import numpy as np
import scipy.sparse as sp

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from tinybig.data.base_data import dataset, dataloader
from tinybig.koala.topology.graph import graph as graph_class
from tinybig.koala.linear_algebra import degree_based_normalize_matrix
from tinybig.util.utility import check_file_existence, download_file_from_github


class graph_dataloader(dataloader):
    """
    A dataloader class for graph-based datasets.

    This class extends the base `dataloader` class to handle graph data, including nodes, links, and associated features.

    Attributes
    ----------
    data_profile: dict
        The data profile containing metadata and download links for the graph dataset.
    graph: graph_class
        The loaded graph structure.

    Methods
    -------
    __init__(data_profile: dict = None, ...)
        Initializes the graph dataloader.
    download_data(data_profile: dict, cache_dir: str = None, file_name: str = None)
        Downloads the graph dataset files.
    load_raw(cache_dir: str, device: str = 'cpu', normalization: bool = True, ...)
        Loads the raw graph data from files.
    save_graph(complete_path: str, graph: graph_class = None)
        Saves the graph structure to a file.
    load_graph(complete_path: str)
        Loads the graph structure from a file.
    get_graph()
        Retrieves the loaded graph structure.
    get_adj(graph: graph_class = None)
        Retrieves the adjacency matrix of the graph.
    load(mode: str = 'transductive', ...)
        Loads the dataset, either in transductive or inductive mode.
    get_train_test_idx(X: torch.Tensor = None, y: torch.Tensor = None, ...)
        Abstract method to generate train and test indices for the dataset.
    """

    def __init__(self, data_profile: dict = None, name: str = 'graph_data', train_batch_size: int = 64, test_batch_size: int = 64):
        """
        Initializes the graph dataloader.

        Parameters
        ----------
        data_profile: dict, optional
            Metadata and download links for the graph dataset.
        name: str, default = 'graph_data'
            The name of the dataloader instance.
        train_batch_size: int, default = 64
            Batch size for the training dataset.
        test_batch_size: int, default = 64
            Batch size for the testing dataset.

        Returns
        -------
        None
        """
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

        self.data_profile = data_profile
        self.graph = None

    @staticmethod
    def download_data(data_profile: dict, cache_dir: str = None, file_name: str = None):
        """
        Downloads the graph dataset files.

        Parameters
        ----------
        data_profile: dict
            Metadata and download links for the graph dataset.
        cache_dir: str, optional
            Directory to store the downloaded files. Defaults to './data/'.
        file_name: str, optional
            Specific file name to download. If None, all files in the `data_profile` are downloaded.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `data_profile` is None or doesn't contain the 'url' key.
        """
        if data_profile is None:
            raise ValueError('The data profile must be provided.')

        if cache_dir is None:
            cache_dir = './data/'

        if data_profile is None or 'url' not in data_profile:
            raise ValueError('data_profile must not be None and should contain "url" key...')

        if file_name is None:
            for file_name in data_profile['url']:
                download_file_from_github(url_link=data_profile['url'][file_name], destination_path="{}/{}".format(cache_dir, file_name))
        else:
            assert file_name in data_profile['url']
            download_file_from_github(url_link=data_profile['url'][file_name], destination_path="{}/{}".format(cache_dir, file_name))


    def load_raw(self, cache_dir: str, device: str = 'cpu', normalization: bool = True, normalization_mode: str = 'row'):
        """
        Loads the raw graph data from files.

        Parameters
        ----------
        cache_dir: str
            Directory containing the graph data files.
        device: str, default = 'cpu'
            Device to store the data.
        normalization: bool, default = True
            Whether to normalize the node features.
        normalization_mode: str, default = 'row'
            Mode of normalization ('row' or 'column').

        Returns
        -------
        tuple
            The graph structure, node features (X), and labels (y).

        Raises
        ------
        FileNotFoundError
            If the required files are not found in the cache directory.
        """
        if not check_file_existence("{}/node".format(cache_dir)):
            self.download_data(data_profile=self.data_profile, cache_dir=cache_dir, file_name='node')
        if not check_file_existence("{}/link".format(cache_dir)):
            self.download_data(data_profile=self.data_profile, cache_dir=cache_dir, file_name='link')

        idx_features_labels = np.genfromtxt("{}/node".format(cache_dir), dtype=np.dtype(str))
        X = torch.tensor(sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32).todense())
        y = dataloader.encode_str_labels(labels=idx_features_labels[:, -1], one_hot=False)

        if normalization:
            X = degree_based_normalize_matrix(mx=X, mode=normalization_mode)

        nodes = np.array(idx_features_labels[:, 0], dtype=np.int32).tolist()
        links = np.genfromtxt("{}/link".format(cache_dir), dtype=np.int32).tolist()
        graph = graph_class(
            nodes=nodes, links=links, directed=True, device=device
        )
        return graph, X, y

    def save_graph(self, complete_path: str, graph: graph_class = None):
        """
        Saves the graph structure to a file.

        Parameters
        ----------
        complete_path: str
            The file path to save the graph structure.
        graph: graph_class, optional
            The graph structure to save. If None, the internal graph is used.

        Returns
        -------
        str
            The path to the saved graph file.

        Raises
        ------
        ValueError
            If no graph structure is loaded or the path is not provided.
        """
        graph = graph if graph is not None else self.graph
        if graph is None:
            raise ValueError('The graph structure has not been loaded yet...')
        if complete_path is None:
            raise ValueError('The cache complete_path has not been set yet...')
        return graph.save(complete_path=complete_path)

    def load_graph(self, complete_path: str):
        """
        Loads the graph structure from a file.

        Parameters
        ----------
        complete_path: str
            The file path to load the graph structure from.

        Returns
        -------
        graph_class
            The loaded graph structure.

        Raises
        ------
        ValueError
            If the file path is not provided.
        """
        if complete_path is None:
            raise ValueError('The cache complete_path has not been set yet...')
        self.graph = graph_class.load(complete_path=complete_path)
        return self.graph

    def get_graph(self):
        """
        Retrieves the loaded graph structure.

        Returns
        -------
        graph_class
            The loaded graph structure, or None if no graph is loaded.
        """
        return self.graph

    def get_adj(self, graph: graph_class = None):
        """
        Retrieves the adjacency matrix of the graph.

        Parameters
        ----------
        graph: graph_class, optional
            The graph structure to use. If None, the internal graph is used.

        Returns
        -------
        torch.Tensor
            The adjacency matrix of the graph.

        Raises
        ------
        ValueError
            If no graph structure is loaded.
        """
        graph = graph if graph is not None else self.graph
        if graph is None:
            raise ValueError('The graph structure has not been loaded yet...')
        return graph.to_matrix(
            normalization=True,
            normalization_mode='row',
        )

    def load(self, mode: str = 'transductive', cache_dir: str = None, device: str = 'cpu',
             train_percentage: float = 0.5, random_state: int = 1234, shuffle: bool = False, *args, **kwargs):
        """
        Loads the dataset in either transductive or inductive mode.

        Parameters
        ----------
        mode: str, default = 'transductive'
            Mode of loading the dataset ('transductive' or 'inductive').
        cache_dir: str, optional
            Directory containing the graph data files. Defaults to './data/{name}'.
        device: str, default = 'cpu'
            Device to store the data.
        train_percentage: float, default = 0.5
            Percentage of data to use for training in inductive mode.
        random_state: int, default = 1234
            Seed for random number generation.
        shuffle: bool, default = False
            Whether to shuffle the data.

        Returns
        -------
        dict
            A dictionary containing train/test loaders and graph structure.

        Raises
        ------
        ValueError
            If required files are not found or the graph is not properly loaded.
        """
        cache_dir = cache_dir if cache_dir is not None else "./data/{}".format(self.name)
        self.graph, X, y = self.load_raw(cache_dir=cache_dir, device=device)

        if mode == 'transductive':
            warnings.warn("For transductive settings, the train, test, and val partition will not follow the provided parameters (e.g., train percentage, batch size, etc.)...")
            train_idx, test_idx = self.get_train_test_idx(X=X, y=y)
            complete_dataset = dataset(X, y)
            complete_dataloader = DataLoader(dataset=complete_dataset, batch_size=len(X), shuffle=False)
            return {'train_idx': train_idx, 'test_idx': test_idx, 'train_loader': complete_dataloader, 'test_loader': complete_dataloader, 'graph_structure': self.graph}
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=int(train_percentage * len(X)),
                random_state=random_state, shuffle=shuffle
            )
            train_dataset = dataset(X_train, y_train)
            test_dataset = dataset(X_test, y_test)
            if self.train_batch_size >= 1:
                train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True)
            else:
                train_loader = DataLoader(dataset=train_dataset, batch_size=len(X_train), shuffle=True)
            if self.test_batch_size >= 1:
                test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batch_size, shuffle=False)
            else:
                test_loader = DataLoader(dataset=test_dataset, batch_size=len(X_test), shuffle=False)
            return {'train_loader': train_loader, 'test_loader': test_loader, 'graph_structure': self.graph}

    @abstractmethod
    def get_train_test_idx(self, X: torch.Tensor = None, y: torch.Tensor = None, *args, **kwargs):
        """
        Abstract method to generate train and test indices for the dataset.

        Parameters
        ----------
        X: torch.Tensor, optional
            Node features.
        y: torch.Tensor, optional
            Labels.

        Returns
        -------
        tuple
            Train and test indices.
        """
        pass


CORA_DATA_PROFILE = {
    'name': 'Cora',
    'node_number': 2708,
    'link_number': 10556,
    'feature_number': 1433,
    'class_number': 7,
    'url': {
        'node': "https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/graph/cora/node",
        'link': "https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/graph/cora/link",
    }
}

CITESEER_DATA_PROFILE = {
    'name': 'Citeseer',
    'node_number': 3327,
    'link_number': 9104,
    'feature_number': 3703,
    'class_number': 6,
    'url': {
        'node': "https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/graph/citeseer/node",
        'link': "https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/graph/citeseer/link",
    }
}

PUBMED_DATA_PROFILE = {
    'name': 'Pubmed',
    'node_number': 19717,
    'link_number': 88648,
    'feature_number': 500,
    'class_number': 3,
    'url': {
        'node': "https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/graph/pubmed/node",
        'link': "https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/graph/pubmed/link",
    }
}


class cora(graph_dataloader):
    """
    A dataloader class for the Cora dataset.

    This class extends `graph_dataloader` to handle the Cora graph dataset,
    which is commonly used in graph-based machine learning research.

    Attributes
    ----------
    data_profile: dict
        Metadata and download links specific to the Cora dataset.
    graph: graph_class
        The loaded graph structure for the Cora dataset.

    Methods
    -------
    __init__(name: str = 'cora', train_batch_size: int = 64, test_batch_size: int = 64, ...)
        Initializes the dataloader for the Cora dataset.
    get_train_test_idx(X: torch.Tensor = None, y: torch.Tensor = None, ...)
        Generates train and test indices for the Cora dataset.
    """
    def __init__(self, name: str = 'cora', train_batch_size: int = 64, test_batch_size: int = 64, *args, **kwargs):
        """
        Initializes the dataloader for the Cora dataset.

        Parameters
        ----------
        name: str, default = 'cora'
            The name of the dataset.
        train_batch_size: int, default = 64
            Batch size for the training dataset.
        test_batch_size: int, default = 64
            Batch size for the testing dataset.

        Returns
        -------
        None
        """
        super().__init__(data_profile=CORA_DATA_PROFILE, name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    def get_train_test_idx(self, X: torch.Tensor = None, y: torch.Tensor = None, *args, **kwargs):
        """
        Generates train and test indices for the Cora dataset.

        Parameters
        ----------
        X: torch.Tensor, optional
            Node features (not used in this method).
        y: torch.Tensor, optional
            Labels (not used in this method).

        Returns
        -------
        tuple
            Train indices (`torch.LongTensor`) and test indices (`torch.LongTensor`).

        Notes
        -----
        The train indices are predefined as the first 140 nodes.
        The test indices are predefined as nodes 500 to 1499.
        """
        train_idx = torch.LongTensor(range(140))
        test_idx = torch.LongTensor(range(500, 1500))
        return train_idx, test_idx


class citeseer(graph_dataloader):
    """
    A dataloader class for the Citeseer dataset.

    This class extends `graph_dataloader` to handle the Citeseer graph dataset,
    which is another benchmark dataset for graph-based machine learning tasks.

    Attributes
    ----------
    data_profile: dict
        Metadata and download links specific to the Citeseer dataset.
    graph: graph_class
        The loaded graph structure for the Citeseer dataset.

    Methods
    -------
    __init__(name: str = 'citeseer', train_batch_size: int = 64, test_batch_size: int = 64, ...)
        Initializes the dataloader for the Citeseer dataset.
    get_train_test_idx(X: torch.Tensor = None, y: torch.Tensor = None, ...)
        Generates train and test indices for the Citeseer dataset.
    """
    def __init__(self, name: str = 'citeseer', train_batch_size: int = 64, test_batch_size: int = 64, *args, **kwargs):
        """
        Initializes the dataloader for the Citeseer dataset.

        Parameters
        ----------
        name: str, default = 'citeseer'
            The name of the dataset.
        train_batch_size: int, default = 64
            Batch size for the training dataset.
        test_batch_size: int, default = 64
            Batch size for the testing dataset.

        Returns
        -------
        None
        """
        super().__init__(data_profile=CITESEER_DATA_PROFILE, name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    def get_train_test_idx(self, X: torch.Tensor = None, y: torch.Tensor = None, *args, **kwargs):
        """
        Generates train and test indices for the Citeseer dataset.

        Parameters
        ----------
        X: torch.Tensor, optional
            Node features (not used in this method).
        y: torch.Tensor, optional
            Labels (not used in this method).

        Returns
        -------
        tuple
            Train indices (`torch.LongTensor`) and test indices (`torch.LongTensor`).

        Notes
        -----
        The train indices are predefined as the first 120 nodes.
        The test indices are predefined as nodes 200 to 1199.
        """
        train_idx = torch.LongTensor(range(120))
        test_idx = torch.LongTensor(range(200, 1200))
        return train_idx, test_idx


class pubmed(graph_dataloader):
    """
    A dataloader class for the PubMed dataset.

    This class extends `graph_dataloader` to handle the PubMed graph dataset,
    a large-scale dataset used for graph-based learning.

    Attributes
    ----------
    data_profile: dict
        Metadata and download links specific to the PubMed dataset.
    graph: graph_class
        The loaded graph structure for the PubMed dataset.

    Methods
    -------
    __init__(name: str = 'pubmed', train_batch_size: int = 64, test_batch_size: int = 64, ...)
        Initializes the dataloader for the PubMed dataset.
    get_train_test_idx(X: torch.Tensor = None, y: torch.Tensor = None, ...)
        Generates train and test indices for the PubMed dataset.
    """
    def __init__(self, name: str = 'pubmed', train_batch_size: int = 64, test_batch_size: int = 64, *args, **kwargs):
        """
        Initializes the dataloader for the PubMed dataset.

        Parameters
        ----------
        name: str, default = 'pubmed'
            The name of the dataset.
        train_batch_size: int, default = 64
            Batch size for the training dataset.
        test_batch_size: int, default = 64
            Batch size for the testing dataset.

        Returns
        -------
        None
        """
        super().__init__(data_profile=PUBMED_DATA_PROFILE, name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    def get_train_test_idx(self, X: torch.Tensor = None, y: torch.Tensor = None, *args, **kwargs):
        """
        Generates train and test indices for the PubMed dataset.

        Parameters
        ----------
        X: torch.Tensor, optional
            Node features (not used in this method).
        y: torch.Tensor, optional
            Labels (not used in this method).

        Returns
        -------
        tuple
            Train indices (`torch.LongTensor`) and test indices (`torch.LongTensor`).

        Notes
        -----
        The train indices are predefined as the first 60 nodes.
        The test indices are predefined as nodes 6300 to 7299.
        """
        train_idx = torch.LongTensor(range(60))
        test_idx = torch.LongTensor(range(6300, 7300))
        return train_idx, test_idx

