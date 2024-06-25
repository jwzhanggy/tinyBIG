# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod
from torch.utils.data import Dataset


class dataloader:
    def __init__(self, name='base_dataloader', *args, **kwargs):
        self.name = name

    @abstractmethod
    def load(self, *args, **kwargs):
        pass


class dataset_template(Dataset):
    def __init__(self, X, y, *args, **kwargs):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx, *args, **kwargs):
        sample = self.X[idx]
        target = self.y[idx]
        return sample, target
