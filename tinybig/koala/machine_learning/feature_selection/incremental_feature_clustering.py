
# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Incremental Feature Clustering #
##################################

from typing import Union

import numpy as np
import torch

from sklearn.cluster import SpectralClustering

from tinybig.koala.machine_learning.feature_selection import feature_selection
from tinybig.koala.linear_algebra import euclidean_distance, batch_euclidean_distance


class incremental_feature_clustering(feature_selection):

    def __init__(self, name: str = 'incremental_variance_threshold', random_state: int = 42, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.random_state = random_state
        self.feature_clustering_model = SpectralClustering(n_clusters=self.n_feature, affinity='precomputed', random_state=self.random_state)

        self.D = None
        self.t = None

    def update_n_feature(self, new_n_feature: int):
        self.set_n_feature(new_n_feature)
        self.feature_clustering_model = SpectralClustering(n_clusters=new_n_feature, affinity='precomputed', random_state=self.random_state)

    def update_D(self, new_D: torch.Tensor):
        if self.incremental:
            if self.D is None:
                self.D = torch.zeros_like(new_D)
                self.t = 0

            assert new_D.shape == self.D.shape and self.t >= 0
            self.t += 1
            old_D = self.D
            self.D = ((self.t - 1) * self.D + new_D)/self.t

            if self.t >= self.t_threshold or euclidean_distance(x=old_D.reshape(-1), x2=self.D.reshape(-1)) < self.incremental_stop_threshold:
                self.incremental = False
        else:
            self.D = new_D

    def fit(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        new_D = batch_euclidean_distance(torch.tensor(X))
        self.update_D(new_D)
        sigma = 1.0
        similarity_matrix = np.exp(-self.D ** 2 / (2. * sigma ** 2))
        if isinstance(similarity_matrix, torch.Tensor):
            similarity_matrix = similarity_matrix.detach().cpu().numpy()
        self.feature_clustering_model.fit(similarity_matrix)

    def compute_centroids(self, X: Union[np.ndarray, torch.Tensor], labels: np.array, n_clusters: int):
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()
        else:
            input_X = X

        centroids = np.zeros((input_X.shape[0], n_clusters))
        for i in range(n_clusters):
            points = input_X[:,labels == i]
            if points.shape[1] > 0:
                centroids[:,i] = np.mean(points, axis=1)
        return torch.tensor(centroids) if isinstance(X, torch.Tensor) and not isinstance(centroids, torch.Tensor) else centroids

    def transform(self, X: Union[np.ndarray, torch.Tensor], device: str = 'cpu', *args, **kwargs):
        if isinstance(X, torch.Tensor):
            input_X = X.detach().cpu().numpy()
        else:
            input_X = X

        assert self.D is not None and self.D.shape == (input_X.shape[1], input_X.shape[1])
        assert self.n_feature is not None and 0 <= self.n_feature <= input_X.shape[1]

        labels = self.feature_clustering_model.labels_
        X_selected = self.compute_centroids(X=input_X, labels=labels, n_clusters=self.n_feature)

        assert X_selected.shape[1] == self.n_feature
        return torch.tensor(X_selected) if isinstance(X, torch.Tensor) and not isinstance(X_selected, torch.Tensor) else X_selected





