# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
The forward-forward algorithm based learner.

This module implements the forward-forward learner, which can also be used
to train, validate and test the RPN model within the tinyBIG toolkit.
"""
from tinybig.learner.base_learner import learner


# TBD
class forward_learner(learner):
    """
    The forward-forward algorithm based learner algorithm.

    This algorithm is still under testing and tuning of its effectiveness and efficiency on training RPN.
    This class will be updated when the algorithm is mature and ready to be used
    """
    def __init__(self, name='forward_forward_algorithm'):
        super().__init__(name=name)

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass



