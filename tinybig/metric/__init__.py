
"""
This module provides the evaluation metrics that can be used with the {{our}} model.

The module contains the following metric classes:
- metric: The base evaluation metric defining the class template
- accuracy: The Accuracy metric for classification task result evaluation.
- f1: The F1 metric for classification task result evaluation.
- mse: The MSE (Mean Squared Error) for regression task result evaluation.
"""

from tinybig.metric.metric import metric
from tinybig.metric.metric2 import metric2
from tinybig.metric.classification_metric import accuracy, f1
from tinybig.metric.regression_metric import mse
