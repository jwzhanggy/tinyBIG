# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
The classification evaluation module.


"""

from tinybig.metric.metric import metric

from sklearn.metrics import accuracy_score, f1_score


class accuracy(metric):
    """
    class accuracy

    The accuracy evaluation metric. It inherits from the base metric class.

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
        It performs the initialization of the evaluation metric. Its internal metric calculation method is declared to be accuracy_score from sklearn.

    evaluate
        It implements the abstract evaluate method declared in the base metric class. The method performs the evaluation based on the inputs.

    __call__
        It reimplements the abstract callable method declared in the base metric class.

    """
    def __init__(self, name='accuracy'):
        """
        The initialization method of the accuracy evaluation metric.

        It initializes an accuracy evaluation metric object based on the input metric name. The method will also call the initialization method of the base class as well.

        Parameters
        ----------
        name: str, default = 'accuracy'
            The name of the evaluation metric.
        """
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