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
    """
        A base class to represent a geometric space with a specified center, universe number,
        distance metric, and associated coordinates.

        This class defines the basic structure for geometric spaces and provides
        methods for calculating distances, managing coordinates, and updating
        spatial properties. It is meant to be extended for specific geometric implementations.

        Attributes
        ----------
        name : str
            The name of the geometric space.
        center : coordinate
            The center point of the geometric space.
        universe_num : int
            The number of universe spaces in the geometric structure.
        distance_metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            A callable function to compute the distance between two points.
        device : str
            The device used for computation, such as 'cpu' or 'cuda'.
        coordinates : list[coordinate]
            A list of coordinates defining the space.

        Methods
        -------
        calculate_distance(coord1, universe1, coord2, universe2)
            Calculates the distance between two coordinates within the same universe.
        get_volume(across_universe)
            Returns the volume (number of points) of the geometric space.
        get_center()
            Returns the center of the geometric space.
        get_universe_num()
            Returns the number of universes in the geometric space.
        get_coordinates()
            Returns all the coordinates in the geometric space.
        update_center(new_center)
            Updates the center of the space and regenerates coordinates.
        get_relative_coordinates(center)
            Calculates the relative coordinates with respect to a given center.
        generate_coordinates(*args, **kwargs)
            Abstract method for generating coordinates; must be implemented by subclasses.
    """

    def __init__(
        self,
        center: coordinate,
        universe_num: int = 1,
        name: str = 'base_geometry',
        distance_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = euclidean_distance,
        device: str = 'cpu',
        *args, **kwargs
    ):
        """
            Initializes the geometric space with a center, universes, and distance metric.

            Parameters
            ----------
            center : coordinate
                The central point of the geometric space.
            universe_num : int, optional
                The number of universes in the space. Defaults to 1.
            name : str, optional
                The name of the geometric space. Defaults to 'base_geometry'.
            distance_metric : Callable, optional
                The metric function to calculate distances between points.
                Defaults to euclidean_distance.
            device : str, optional
                The device for computation, such as 'cpu' or 'cuda'. Defaults to 'cpu'.
            *args, **kwargs
                Additional parameters for future extensibility.

            Raises
            ------
            ValueError
                If `universe_num` is less than or equal to 0.

            Attributes
            ----------
            name : str
                The name of the geometric space.
            center : coordinate
                The central coordinate of the space.
            universe_num : int
                The number of universes in the space.
            distance_metric : Callable
                The function used to calculate distances between coordinates.
            device : str
                The computation device for the space.
            coordinates : list[coordinate]
                A list of all coordinates in the geometric space.
        """

        if universe_num <= 0:
            raise ValueError('universe_num must be greater than 0...')
        self.name = name
        self.center = center
        self.universe_num = universe_num
        self.distance_metric = distance_metric
        self.device = device
        self.coordinates = self.generate_coordinates()

    def calculate_distance(self, coord1: coordinate, universe1: int, coord2: coordinate, universe2: int) -> torch.Tensor:
        """
            Calculates the distance between two coordinates within the same universe.

            Parameters
            ----------
            coord1 : coordinate
                The first coordinate.
            universe1 : int
                The universe ID for the first coordinate.
            coord2 : coordinate
                The second coordinate.
            universe2 : int
                The universe ID for the second coordinate.

            Returns
            -------
            torch.Tensor
                The computed distance between `coord1` and `coord2`.

            Raises
            ------
            ValueError
                If either coordinate is not in the current space, if the universe IDs
                are invalid, or if the coordinates belong to different universes.
        """

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
        """
            Returns the volume (number of points) of the geometric space.

            Parameters
            ----------
            across_universe : bool, optional
                If True, includes all universes in the volume calculation. Defaults to False.

            Returns
            -------
            int
                The total number of points in the space.
        """

        if across_universe:
            return len(self.coordinates) * self.universe_num
        else:
            return len(self.coordinates)

    def get_center(self):
        """
            Returns the center of the geometric space.

            Returns
            -------
            coordinate
                The center point of the space.
        """
        return self.center

    def get_universe_num(self):
        """
            Returns the number of universes in the geometric space.

            Returns
            -------
            int
                The number of universes.
        """
        return self.universe_num

    def get_coordinates(self):
        """
            Returns all the coordinates in the geometric space.

            Returns
            -------
            list[coordinate]
                A list of all coordinates in the space.
        """
        return self.coordinates

    def update_center(self, new_center: coordinate):
        """
            Updates the center of the geometric space and regenerates coordinates.

            Parameters
            ----------
            new_center : coordinate
                The new center point.

            Raises
            ------
            AssertionError
                If the dimensions of the new center do not match the current center.
        """
        assert self.center.dimension() == new_center.dimension()
        self.center = new_center
        self.coordinates = self.generate_coordinates()

    def get_relative_coordinates(self, center: coordinate):
        """
            Calculates the relative coordinates with respect to a given center.

            Parameters
            ----------
            center : coordinate
                The reference center point.

            Returns
            -------
            dict
                A dictionary mapping relative coordinates to their occurrences.
        """
        relative_coord = center - self.center
        relative_coordinates = {}
        for coord in self.coordinates:
            relative_coordinates[coord + relative_coord] = 1
        return relative_coordinates

    @abstractmethod
    def generate_coordinates(self, *args, **kwargs):
        """
            Abstract method for generating the coordinates of the geometric space.

            Must be implemented by subclasses.

            Parameters
            ----------
            *args, **kwargs
                Additional parameters for coordinate generation.

            Returns
            -------
            list[coordinate]
                The generated coordinates.
        """
        pass

