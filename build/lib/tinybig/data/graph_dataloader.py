# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from torch.utils.data import DataLoader

from tinybig.data.base_data import dataloader

#https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid

class cora(dataloader):
    pass

class citeseer(dataloader):
    pass

class pubmed(dataloader):
    pass