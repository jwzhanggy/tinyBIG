"""
This module defines several frequently used dataset, which can be used for training the RPN model in the tinyBIG toolkit.

## Organization of this Module

This module contains the following categories of learner classes:

* Base Data
* Function Data
* Image Data
* Text Data
* Tabular Data
"""


from .base_data import (
    dataloader,
    dataset_template
)

from .vision_dataloader import (
    mnist,
    cifar10,
    imagenet
)

from .text_dataloader_torchtext import (
    text_dataloader
)
from .text_dataloader_torchtext import (
    imdb,
    sst2,
    agnews
)

from .graph_dataloader import (
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
