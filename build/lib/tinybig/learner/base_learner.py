# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

"""
The base learner class.

This module implements the base learner class, which can be used as a
learner template for defining various learning algorithms.
"""
from abc import abstractmethod


class learner:
    """
    The base learner class.

    This base class defines a base learner template. It can be used to define several learning algorithms that can be
    applied to train the RPN model in the tinyBIG toolkit.

    Attributes
    ----------
    name: str, default = 'base_learner'
        Name of the base learner class.

    Methods
    ----------
    __init__
        The base learner initialization method

    train
        The training method of the base learner on the training set.

    test
        The testing method of the base learner on the testing set.
    """
    def __init__(self, name='base_learner', *args, **kwargs):
        self.name = name

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass


