# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
Classification result evaluation metrics.

This module contains the evaluation metrics for classification results, including accuracy and f1.
"""

from tinybig.metric.base_metric import metric

from sklearn.metrics import accuracy_score, f1_score


class accuracy(metric):
    """
    The accuracy evaluation metric.

    The class inherits from the base metric class.

    ...

    Attributes
    ----------
    name: str, default = 'accuracy'
        Name of the accuracy evaluation metric.
    metric: object
        The accuracy evaluation metric calculation method.

    Methods
    ----------
    __init__
        It performs the initialization of the accuracy evaluation metric. Its internal metric calculation method is declared to be accuracy_score from sklearn.

    evaluate
        It implements the abstract evaluate method declared in the base metric class. The method calculates the accuracy score of the inputs.

    __call__
        It reimplements the abstract callable method declared in the base metric class.

    """
    def __init__(self, name: str = 'accuracy'):
        """
        The initialization method of the accuracy evaluation metric.

        It initializes an accuracy evaluation metric object based on the input metric name.
        This method will also call the initialization method of the base class as well.
        The metric calculation approach is initialized as the sklearn.metrics.accuracy_score.

        Parameters
        ----------
        name: str, default = 'accuracy'
            The name of the evaluation metric.
        """
        super().__init__(name=name)
        self.metric = accuracy_score

    def evaluate(self, y_true: list, y_pred: list, *args, **kwargs):
        """
        The evaluate method of the accuracy evaluation metric class.

        It calculates the accuracy scores based on the provided input parameters "y_true" and "y_pred".
        The method will return calculated accuracy score as the output.

        Examples
        ----------
        >>> from tinybig.metric import accuracy as accuracy_metric
        >>> acc_metric = accuracy_metric(name='accuracy_metric')
        >>> y_true = [1, 1, 0, 0]
        >>> y_pred = [1, 1, 0, 1]
        >>> acc_metric.evaluate(y_pred=y_pred, y_true=y_true)
        0.75

        Parameters
        ----------
        y_true: list
            The list of true labels of data instances.
        y_pred: list
            The list of predicted labels of data instances.
        args: list
            Other parameters
        kwargs: dict
            Other parameters

        Returns
        -------
        float | list
            The calculated accuracy score of the input parameters.
        """
        return self.metric(y_true=y_true, y_pred=y_pred)

    def __call__(self, y_true: list, y_pred: list, *args, **kwargs):
        """
        The callable method of the accuracy metric class.

        It re-implements the build-in callable method.
        This method will call the evaluate method to calculate the accuracy of the input parameters.

        Examples
        ----------
        >>> from tinybig.metric import accuracy as accuracy_metric
        >>> acc_metric = accuracy_metric(name='accuracy_metric')
        >>> y_true = [1, 1, 0, 0]
        >>> y_pred = [1, 1, 0, 1]
        >>> acc_metric(y_pred=y_pred, y_true=y_true)
        0.75

        Parameters
        ----------
        y_true: list
            The list of true labels of data instances.
        y_pred: list
            The list of predicted labels of data instances.
        args: list
            Other parameters
        kwargs: dict
            Other parameters

        Returns
        -------
        float | list
            The calculated accuracy score of the input parameters.
        """
        return self.evaluate(y_true=y_true, y_pred=y_pred, *args, **kwargs)


class f1(metric):
    """
    The f1 evaluation metric.

    The class inherits from the base metric class.

    ...

    Attributes
    ----------
    name: str, default = 'f1'
        Name of the accuracy evaluation metric.
    metric: object
        The accuracy evaluation metric calculation method.
    average: str, default = 'binary'
        The average parameter used for the metric calculation. It takes value from {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None, default=’binary’

    Methods
    ----------
    __init__
        It performs the initialization of the f1 evaluation metric. Its internal metric calculation method is declared to be f1_score from sklearn.

    evaluate
        It implements the abstract evaluate method declared in the base metric class. The method calculates the f1 score of the input prediction labels.

    __call__
        It reimplements the abstract callable method declared in the base metric class.

    """
    def __init__(self, name: str = 'f1', average: str = 'binary'):
        """
        The initialization method of the f1 evaluation metric.

        It initializes an f1 evaluation metric object based on the input metric name.
        This method will also call the initialization method of the base class as well.
        The metric calculation approach is initialized as the sklearn.metrics.f1_score with the default average parameter "binary".

        Parameters
        ----------
        name: str, default = 'f1'
            The name of the evaluation metric.
        average: str, default = 'binary'
            The average parameter of the f1 evaluation metric.
        """
        super().__init__(name=name)
        self.metric = f1_score
        self.average = average

    def evaluate(self, y_true: list, y_pred: list, average=None, *args, **kwargs):
        """
        The evaluate method of the f1 evaluation metric class.

        It calculates the accuracy scores based on the provided input parameters "y_true" and "y_pred".
        The method will return calculated f1 score as the output.

        Examples
        ----------
        Binary classification f1 score
        >>> from tinybig.metric import f1 as f1_metric
        >>> y_true = [1, 1, 0, 0]
        >>> y_pred = [1, 1, 0, 1]
        >>> f1_metric = f1_metric(name='f1_metric', average='binary')
        >>> f1_metric.evaluate(y_true=y_true, y_pred=y_pred)
        0.8

        Multi-class Classification f1 score
        >>> y_true = [0, 1, 2, 0, 1, 2]
        >>> y_pred = [0, 2, 1, 0, 0, 1]
        >>> f1_metric_macro = f1_metric(name='f1_metric_macro', average='macro')
        >>> f1_metric_macro.evaluate(y_pred=y_pred, y_true=y_true)
        0.26...
        >>> f1_metric_micro = f1_metric(name='f1_metric_micro', average='micro')
        >>> f1_metric_micro.evaluate(y_true=y_true, y_pred=y_pred)
        0.33...
        >>> f1_metric_micro = f1_metric(name='f1_metric_micro', average='micro')
        >>> f1_metric_micro.evaluate(y_true=y_true, y_pred=y_pred)
        0.26...
        >>> f1_metric = f1_metric(name='f1_metric', average=None)
        >>> f1_metric.evaluate(y_true=y_true, y_pred=y_pred)
        array([0.8, 0. , 0. ])

        Multi-label classification f1 score
        >>> y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
        >>> y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
        >>> f1_metric = f1_metric(name='f1_metric', average=None)
        >>> f1_metric.evaluate(y_true=y_true, y_pred=y_pred)
        array([0.66666667, 1.        , 0.66666667])

        Parameters
        ----------
        y_true: list
            The list of true labels of data instances.
        y_pred: list
            The list of predicted labels of data instances.
        args: list
            Other parameters
        kwargs: dict
            Other parameters

        Returns
        -------
        float | list
            The calculated f1 score of the input parameters.
        """
        average = average if average is not None else self.average
        return self.metric(y_true=y_true, y_pred=y_pred, average=average)

    def __call__(self, y_true: list, y_pred: list, *args, **kwargs):
        """
        The callable method of the f1 metric class.

        It re-implements the build-in callable method.
        This method will call the evaluate method to calculate the f1 of the input parameters.

        Examples
        ----------
        Binary classification f1 score
        >>> from tinybig.metric import f1 as f1_metric
        >>> y_pred = [1, 1, 0, 0]
        >>> y_true = [1, 1, 0, 1]
        >>> f1_metric = f1_metric(name='f1_metric', average='binary')
        >>> f1_metric(y_true=y_true, y_pred=y_pred)
        0.8

        Parameters
        ----------
        y_true: list
            The list of true labels of data instances.
        y_pred: list
            The list of predicted labels of data instances.
        args: list
            Other parameters
        kwargs: dict
            Other parameters

        Returns
        -------
        float | list
            The calculated f1 score of the input parameters.
        """
        return self.evaluate(y_true=y_true, y_pred=y_pred, *args, **kwargs)


if __name__ == '__main__':
    print(accuracy.__doc__)
    help(f1)