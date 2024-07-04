# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
Regression result evaluation metrics.

This module contains the evaluation metrics for regression results, including mse (mean squared error).
"""

from sklearn.metrics import mean_squared_error
from tinybig.metric.base_metric import metric


class mse(metric):
    """
    The mse evaluation metric.

    The class inherits from the base metric class.

    ...

    Attributes
    ----------
    name: str, default = 'mean_squared_error'
        Name of the mse evaluation metric.
    metric: object
        The accuracy evaluation metric calculation method.

    Methods
    ----------
    __init__
        It performs the initialization of the mse evaluation metric. Its internal metric calculation method is declared to be mean_squared_error from sklearn.

    evaluate
        It implements the abstract evaluate method declared in the base metric class. The method calculates the mse score of the inputs.

    __call__
        It reimplements the abstract callable method declared in the base metric class.

    """
    def __init__(self, name='mean_squared_error'):
        """
        The initialization method of the mse evaluation metric.

        It initializes a mse evaluation metric object based on the input metric name.
        This method will also call the initialization method of the base class as well.
        The metric calculation approach is initialized as the sklearn.metrics.mean_squared_error.

        Parameters
        ----------
        name: str, default = 'mean_squared_error'
            The name of the evaluation metric.
        """
        super().__init__(name=name)
        self.metric = mean_squared_error

    def evaluate(self, y_true: list, y_score: list, *args, **kwargs):
        """
        The evaluate method of the mse evaluation metric class.

        It calculates the mse scores based on the provided input parameters "y_true" and "y_score".
        The method will return calculated accuracy score as the output.

        Examples
        ----------
        >>> from tinybig.metric import mse as mse_metric
        >>> mse_metric = mse_metric(name='mse_metric')
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_score = [2.5, 0.0, 2, 8]
        >>> mse_metric.evaluate(y_true=y_true, y_score=y_score)
        0.375

        Parameters
        ----------
        y_true: list
            The list of true labels of data instances.
        y_score: list
            The list of predicted scores of data instances.
        args: list
            Other parameters
        kwargs: dict
            Other parameters

        Returns
        -------
        float | list
            The calculated mse score of the input parameters.
        """
        return self.metric(y_true=y_true, y_pred=y_score)

    def __call__(self, y_true: list, y_score: list, *args, **kwargs):
        """
        The callable method of the mse metric class.

        It re-implements the build-in callable method.
        This method will call the evaluate method to calculate the mse of the input parameters.

        Examples
        ----------
        Binary classification f1 score
        >>> from tinybig.metric import mse as mse_metric
        >>> mse_metric = mse_metric(name='mse_metric')
        >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
        >>> y_score = [[0, 2],[-1, 2],[8, -5]]
        >>> mse_metric.evaluate(y_true=y_true, y_score=y_score)
        0.708...

        Parameters
        ----------
        y_true: list
            The list of true labels of data instances.
        y_score: list
            The list of predicted scores of data instances.
        args: list
            Other parameters
        kwargs: dict
            Other parameters

        Returns
        -------
        float | list
            The calculated mse score of the input parameters.
        """
        return self.evaluate(y_true=y_true, y_score=y_score, *args, **kwargs)
