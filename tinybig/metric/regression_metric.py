# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from sklearn.metrics import mean_squared_error
from tinybig.metric.metric import metric


class mse(metric):
    def __init__(self, name='mse'):
        super().__init__(name=name)
        self.metric = mean_squared_error

    def evaluate(self, y_true: list, y_score: list, *args, **kwargs):
        return self.metric(y_true=y_true, y_pred=y_score)

    def __call__(self, y_true: list, y_score: list, *args, **kwargs):
        return self.evaluate(y_true=y_true, y_score=y_score, *args, **kwargs)
