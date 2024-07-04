"""
This module implements the output classes, which can be used for the RPN model output saving and loading.

Specifically, this module will implement a base output class, together with a prediction output class.
Several fundamental output processing, saving and loading functions will be implemented in these classes.

## Organization of this Module

This module contains the following categories of learner classes:

* Base Output Template
* Prediction Output

"""

from tinybig.output.base_output import output
from tinybig.output.prediction_output import prediction
