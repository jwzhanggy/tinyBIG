"""
This module defines the learners that can work for the RPN models in the tinyBIG toolkit.

Specifically, this module will introduce a base learner as a template, based on which,
several other learning algorithms are implemented, including the famous "error back propagation algorithm"
and the recent "forward-forward algorithm".

## Classes in this Module

This module contains the following categories of learner classes:

* Base Learner Template
* Backward Learner
* Forward Learner
"""

from tinybig.learner.base_learner import learner
from tinybig.learner.backward_learner import backward_learner
from tinybig.learner.forward_learner import forward_learner
