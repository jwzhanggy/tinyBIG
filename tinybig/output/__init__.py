"""
This module implements the output classes, which can be used for the RPN model output saving and loading.

Specifically, this module will implement a base output class, together with a prediction output class.
Several fundamental output processing, saving and loading functions will be implemented in these classes.

## Classes in this Module

This module contains the following categories of output processing classes:

* Base Output Template
* RPN Output

"""

from tinybig.output.base_output import output
from tinybig.output.rpn_output import rpn_output
