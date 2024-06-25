# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod


class model:
    def __init__(self, name='model_name', *args, **kwargs):
        self.name = name

    @abstractmethod
    def save_ckpt(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_ckpt(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass