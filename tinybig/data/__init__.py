# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

####################
# tinybig Datasets #
####################

"""
This module defines several frequently used dataset, which can be used for training the RPN model in the tinyBIG toolkit.

## Classes in this Module

This module contains the following categories of dataloader classes:

* Base Dataloader
* Function Dataloaders
* Image Dataloaders
* Text Dataloaders
* Tabular Dataloaders
* Time Series Dataloaders
* Graph Dataloaders
"""


from .base_data import (
    dataloader,
    dataset
)

from .vision_dataloader import (
    vision_dataloader,
    mnist,
    cifar10,
    imagenet
)

from .text_dataloader_torchtext import (
    text_dataloader,
    imdb,
    sst2,
    agnews
)

from .graph_dataloader import (
    graph_dataloader,
    cora,
    citeseer,
    pubmed
)

from .function_dataloader import (
    function_dataloader,
    elementary_function,
    composite_function
)
from .feynman_dataloader import (
    feynman_function,
    dimensionless_feynman_function
)

from .tabular_dataloader import (
    tabular_dataloader,
    diabetes,
    iris,
    banknote
)

from .time_series_dataloader import (
    time_series_dataloader,
    etf,
    stock,
    traffic_la,
    traffic_bay
)

