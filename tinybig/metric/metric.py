# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Base evaluation metric.

This module contains the base evaluation metric class definition.
"""

from abc import abstractmethod


class metric:
    """
    The base class of the evaluation metrics in the tinyBIG toolkit.

    ...

    Attributes
    ----------
    name: str, default = 'base_metric'
        Name of the evaluation metric.

    Methods
    ----------
    __init__
        It performs the initialization of the evaluation metric.

    evaluate
        It performs the evaluation based on the inputs.

    __call__
        It reimplementation the build-in callable method.
    """
    def __init__(self, name: str = 'base_metric', *args, **kwargs):
        """
        The initialization method of the base metric class.

        It initializes a metric object based on the provided method parameters.

        Parameters
        ----------
        name: str, default = 'base_metric'
            The name of the metric, with default value "base_metric".

        Returns
        ----------
        object
            The metric object.
        """
        self.name = name

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        The evaluate method of the base metric class.

        It is declared to be an abstractmethod and needs to be implemented in the inherited evaluation metric classes.
        The evaluate method accepts prediction results and ground-truth results as inputs and return the evaluation metric scores as the outputs.

        Returns
        ----------
        float | dict
            The evaluation metric scores.
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Reimplementation of the build-in callable method.

        It is declared to be an abstractmethod and needs to be implemented in the inherited evaluation metric classes.
        This callable method accepts prediction results and ground-truth results as inputs and return the evaluation metric scores as the outputs.

        Returns
        ----------
        float | dict
            The evaluation metric scores.
        """
        pass


if __name__ == '__main__':
    print(metric.__doc__)
    help(metric)