# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################
# Base Geometry #
#################

from abc import abstractmethod
from typing import Callable

import torch

from tinybig.koala.geometry import coordinate
from tinybig.koala.linear_algebra import euclidean_distance


class geometric_space:
    def __init__(
        self,
        center: coordinate,
        universe_num: int = 1,
        name: str = 'base_geometry',
        distance_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = euclidean_distance,
        device: str = 'cpu',
        *args, **kwargs
    ):
        if universe_num <= 0:
            raise ValueError('universe_num must be greater than 0...')
        self.name = name
        self.center = center
        self.universe_num = universe_num
        self.distance_metric = distance_metric
        self.device = device
        self.coordinates = self.generate_coordinates()

    def calculate_distance(self, coord1: coordinate, universe1: int, coord2: coordinate, universe2: int) -> torch.Tensor:
        if coord1 not in self.coordinates or coord2 not in self.coordinates:
            raise ValueError('coordinates do not exist in the current geometric space...')
        if not (0 <= universe1 < self.universe_num and 0 <= universe2 < self.universe_num):
            raise ValueError('universe ids do not exist in the current geometric space...')
        if universe1 != universe2:
            raise ValueError('the current distance metric can only calculate distance within the same universe space...')
        if self.distance_metric is None:
            raise ValueError('distance_metric must be defined...')

        return self.distance_metric(torch.Tensor(coord1.coords), torch.Tensor(coord2.coords))

    def get_volume(self, across_universe: bool = False):
        if across_universe:
            return len(self.coordinates) * self.universe_num
        else:
            return len(self.coordinates)

    def get_center(self):
        return self.center

    def get_universe_num(self):
        return self.universe_num

    def get_coordinates(self):
        return self.coordinates

    def update_center(self, new_center: coordinate):
        assert self.center.dimension() == new_center.dimension()
        self.center = new_center
        self.coordinates = self.generate_coordinates()

    def get_relative_coordinates(self, center: coordinate):
        relative_coord = center - self.center
        relative_coordinates = {}
        for coord in self.coordinates:
            relative_coordinates[coord + relative_coord] = 1
        return relative_coordinates

    @abstractmethod
    def generate_coordinates(self, *args, **kwargs):
        pass

