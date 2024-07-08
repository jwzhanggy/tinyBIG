
"""
This module provides the evaluation metrics that can be used within the tinyBIG toolkit.


## Classes in this Module

This module contains the following categories of evaluation metric classes:

* metric: The base evaluation metric defining the class template.

* accuracy: The Accuracy metric for classification task result evaluation.

* f1: The F1 metric for classification task result evaluation.

* mse: The MSE (Mean Squared Error) for regression task result evaluation.

"""

from tinybig.metric.base_metric import metric
from tinybig.metric.classification_metric import accuracy, f1
from tinybig.metric.regression_metric import mse
