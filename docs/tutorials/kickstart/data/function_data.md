### Base Dataloaders and Dataset

{{toolkit}} offers two base primitives to work with data: `dataloader` and `dataset`, both defined by module `tinybig.data.base_data` 
in the package. 
`dataset` stores the data instances (including features, labels, and optional encoders for feature embedding), and
`dataloader` wraps an iterable around the `dataset`.

Based on `dataloader` and `dataset`, several dataloaders for specific data modalities have been created:

```python
import tinybig as tb
from tinybig.data import dataloader, dataset
from tinybig.data import function_dataloader, vision_dataloader, text_dataloader, tabular_dataloader
```

Built based on torchvision and torchtext, {{toolkit}} can load many real-world vision data, like MNIST and CIFAR10, and
text data, like IMDB, SST2 and AGNews, for model training and evaluation. In addition, {{toolkit}} also offers a variety
of other well-known datasets by itself, including continuous function datasets, like Elementary, Composite and Feynman functions,
and classic tabular datasets, like Iris, Diabetes and Banknote.