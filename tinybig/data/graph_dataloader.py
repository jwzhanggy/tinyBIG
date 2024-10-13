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
from tinybig.util.utility import check_file_existence, download_file_from_github


class graph_dataloader(dataloader):

    def __init__(self, data_profile: dict = None, name: str = 'graph_data', train_batch_size: int = 64, test_batch_size: int = 64):
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

        self.data_profile = data_profile
        self.graph = None

    @staticmethod
    def download_data(data_profile: dict, cache_dir: str = None, file_name: str = None):
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


    def load_raw(self, cache_dir: str, device: str = 'cpu'):
        if not check_file_existence("{}/node".format(cache_dir)):
            self.download_data(data_profile=self.data_profile, cache_dir=cache_dir, file_name='node')
        if not check_file_existence("{}/link".format(cache_dir)):
            self.download_data(data_profile=self.data_profile, cache_dir=cache_dir, file_name='link')

        idx_features_labels = np.genfromtxt("{}/node".format(cache_dir), dtype=np.dtype(str))
        X = torch.tensor(sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32).todense())
        y = dataloader.encode_str_labels(labels=idx_features_labels[:, -1], one_hot=False)

        nodes = np.array(idx_features_labels[:, 0], dtype=np.int32).tolist()
        links = np.genfromtxt("{}/link".format(cache_dir), dtype=np.int32).tolist()
        graph = graph_class(
            nodes=nodes, links=links, directed=False, device=device
        )
        return graph, X, y

    def save_graph(self, complete_path: str, graph: graph_class = None):
        graph = graph if graph is not None else self.graph
        if graph is None:
            raise ValueError('The graph structure has not been loaded yet...')
        if complete_path is None:
            raise ValueError('The cache complete_path has not been set yet...')
        return graph.save(complete_path=complete_path)

    def load_graph(self, complete_path: str):
        if complete_path is None:
            raise ValueError('The cache complete_path has not been set yet...')
        self.graph = graph_class.load(complete_path=complete_path)
        return self.graph

    def get_graph(self):
        return self.graph

    def get_adj(self, graph: graph_class = None):
        graph = graph if graph is not None else self.graph
        if graph is None:
            raise ValueError('The graph structure has not been loaded yet...')
        return graph.to_matrix(
            normalization=True,
            normalization_mode='row',
        )

    def load(self, mode: str = 'semi_supervised', cache_dir: str = None, device: str = 'cpu',
             train_percentage: float = 0.5, random_state: int = 1234, shuffle: bool = False, *args, **kwargs):

        cache_dir = cache_dir if cache_dir is not None else "./data/{}".format(self.name)
        self.graph, X, y = self.load_raw(cache_dir=cache_dir, device=device)

        if mode == 'semi_supervised':
            warnings.warn("For semi-supervised settings, the train, test, and val partition will not follow the provided parameters (e.g., train percentage, batch size, etc.)...")
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
    def __init__(self, name: str = 'cora', train_batch_size: int = 64, test_batch_size: int = 64, *args, **kwargs):
        super().__init__(data_profile=CORA_DATA_PROFILE, name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    def get_train_test_idx(self, X: torch.Tensor = None, y: torch.Tensor = None, *args, **kwargs):
        train_idx = torch.LongTensor(range(140))
        test_idx = torch.LongTensor(range(200, 1200))
        return train_idx, test_idx


class citeseer(graph_dataloader):
    def __init__(self, name: str = 'citeseer', train_batch_size: int = 64, test_batch_size: int = 64, *args, **kwargs):
        super().__init__(data_profile=CITESEER_DATA_PROFILE, name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    def get_train_test_idx(self, X: torch.Tensor = None, y: torch.Tensor = None, *args, **kwargs):
        train_idx = torch.LongTensor(range(120))
        test_idx = torch.LongTensor(range(200, 1200))
        return train_idx, test_idx


class pubmed(graph_dataloader):
    def __init__(self, name: str = 'pubmed', train_batch_size: int = 64, test_batch_size: int = 64, *args, **kwargs):
        super().__init__(data_profile=PUBMED_DATA_PROFILE, name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    def get_train_test_idx(self, X: torch.Tensor = None, y: torch.Tensor = None, *args, **kwargs):
        train_idx = torch.LongTensor(range(60))
        test_idx = torch.LongTensor(range(6300, 7300))
        return train_idx, test_idx

