# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod


class metric:
    """
    The base class of the evaluation metrics in the {{ toolkit }} toolkit.

    ...

    Attributes
    ----------
        name: str
            name of the evaluation metric

    Methods
    ----------
        evaluate(self, *args, **kwargs):
            It performs the evaluation based on the inputs.

        __call__(self, *args, **kwargs):
            It performs the evaluation based on the inputs by calling the object directly.
    """
    def __init__(self, name: str = 'base_metric', *args, **kwargs):
        """
        Initialization function of the base metric class.

        It can initialize a metric object based on the provided function parameters.

        Parameters
        ----------
        :param name: name of the metric, with default value "base_metric"
        :param args: other arguments.

        Returns
        ----------
        :param kwargs: other arguments.
        """
        self.name = name

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        The evaluate function of the base metric class.

        It is declared to be an abstractmethod and needs to be implemented in the inherited evaluation metric classes.
        The evaluate function accepts prediction results and ground-truth results as inputs and return the evaluation metric scores as the outputs.

        Parameters
        ----------
        :param args: input prediction results and ground-truth results
        :param kwargs: other inputs

        Returns
        ----------
        :return: evaluation metric scores
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Reimplementation of the __call__ function.

        It is declared to be an abstractmethod and needs to be implemented in the inherited evaluation metric classes.

        Parameters
        ----------
        :param args: input prediction results and ground-truth results
        :param kwargs: other inputs

        Returns
        ----------
        :return: evaluation metric scores
        """
        pass


if __name__ == '__main__':
    print(metric.__doc__)
    help(metric)