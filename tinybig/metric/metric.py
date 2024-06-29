# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod


class metric:
    """
    class metric

    The base class of the evaluation metrics in the tinyBIG toolkit.

    Attributes
    ----------
    name: str
        name of the evaluation metric

    Methods
    ----------
    evaluate
        It performs the evaluation based on the inputs.

    __call__
        It performs the evaluation based on the inputs by calling the object directly.
    """
    def __init__(self, name: str = 'base_metric', *args, **kwargs):
        """
        Initialization function of the base metric class.

        It can initialize a metric object based on the provided function parameters.

        Parameters
        ----------
        name
            The name of the metric, with default value "base_metric".

        Returns
        ----------
        The metric object.
        """
        self.name = name

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        The evaluate function of the base metric class.

        It is declared to be an abstractmethod and needs to be implemented in the inherited evaluation metric classes.
        The evaluate function accepts prediction results and ground-truth results as inputs and return the evaluation metric scores as the outputs.

        Returns
        ----------
        The evaluation metric scores.
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Reimplementation of the build-in callable function.

        It is declared to be an abstractmethod and needs to be implemented in the inherited evaluation metric classes.

        Returns
        ----------
        The evaluation metric scores.
        """
        pass


if __name__ == '__main__':
    print(metric.__doc__)
    help(metric)