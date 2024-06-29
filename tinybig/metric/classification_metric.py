# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis


from tinybig.metric.metric import metric

from sklearn.metrics import accuracy_score, f1_score


class accuracy(metric):
    """
    The accuracy evaluation metric.


    """
    def __init__(self, name='accuracy'):
        super().__init__(name=name)
        self.metric = accuracy_score

    def evaluate(self, y_true: list, y_pred: list, *args, **kwargs):
        return self.metric(y_true=y_true, y_pred=y_pred)

    def __call__(self, y_true: list, y_pred: list, *args, **kwargs):
        return self.evaluate(y_true=y_true, y_pred=y_pred, *args, **kwargs)


class f1(metric):
    """
    The f1 evaluation metric.


    """
    def __init__(self, name='f1', average='weighted'):
        super().__init__(name=name)
        self.metric = f1_score
        self.average = average

    def evaluate(self, y_true: list, y_pred: list, average=None, *args, **kwargs):
        average = average if average is not None else self.average
        return self.metric(y_true=y_true, y_pred=y_pred, average=average)

    def __call__(self, y_true: list, y_pred: list, *args, **kwargs):
        return self.evaluate(y_true=y_true, y_pred=y_pred, *args, **kwargs)


if __name__ == '__main__':
    print(accuracy.__doc__)
    help(f1)