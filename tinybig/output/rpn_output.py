# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
The rpn output class.

This class implements the class for the RPN rpn output result saving and loading.
"""

import pickle

from tinybig.output.base_output import output
from tinybig.util.util import create_directory_if_not_exists


class rpn_output(output):
    """
    The RPN model rpn output class.

    It processes the rpn output by the RPN model and handle the output saving and loading.

    Attributes
    ----------
    name: str, default = 'rpn_output'
        Name of the rpn output class object.

    Methods
    ----------
    __init__
        The initialization method of the rpn output class.
    """
    def __init__(self, name='rpn_output'):
        super().__init__(name=name)


