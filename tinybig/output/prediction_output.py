# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis
"""
The prediction output class.

This class implements the class for the RPN prediction output result saving and loading.
"""

import pickle

from tinybig.output.base_output import output
from tinybig.util.util import create_directory_if_not_exists


class prediction(output):
    """
    The RPN model prediction output class.

    It processes the prediction output by the RPN model and handle the output saving and loading.

    Attributes
    ----------
    name: str, default = 'prediction_output'
        Name of the prediction output class object.

    Methods
    ----------
    __init__
        The initialization method of the prediction output class.
    """
    def __init__(self, name='prediction_output'):
        super().__init__(name=name)


