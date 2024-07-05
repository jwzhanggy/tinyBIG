# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod


class visualizer:
    def __init__(self, name='base_visualizer', *args, **kwargs):
        self.name = name

    @abstractmethod
    def plot(self, *args, **kwargs):
        pass
