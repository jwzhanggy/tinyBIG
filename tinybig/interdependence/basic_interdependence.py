# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#########################
# Basic Interdependency #
#########################

import torch
import torch.nn.functional as F

from tinybig.interdependence import interdependence


class constant_interdependence(interdependence):

    def __init__(self, name: str='constant_interdependence', o: int=None, o_prime: int=None, matrix_A: torch.Tensor=None, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.matrix_A = matrix_A
        assert self.matrix_A is not None

        if o is not None:
            assert o == self.matrix_A.size(0)
        self.o = self.matrix_A.size(0)

        if o_prime is not None:
            assert o_prime == self.matrix_A.size(1)
        self.o_prime = self.matrix_A.size(1)

    def calculate_o_prime(self, o: int):
        return self.o_prime

    def forward(self, device='cpu', *args, **kwargs):
        pass