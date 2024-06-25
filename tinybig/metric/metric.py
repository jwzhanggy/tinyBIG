# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod


class metric:
    def __init__(self, name='base_metric', *args, **kwargs):
        self.name = name

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
