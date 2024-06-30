from tinybig.data.dataloader import dataloader, dataset_template
from tinybig.data.vision_dataloader import mnist, cifar10, imagenet
from tinybig.data.text_dataloader_torchtext import text_dataloader, text_dataset_template
from tinybig.data.text_dataloader_torchtext import imdb, sst2, agnews
from tinybig.data.graph_dataloader import cora, citeseer, pubmed
from tinybig.data.feynman_dataloader import feynman_function, dimensionless_feynman_function
from tinybig.data.function_dataloader import elementary_function, composite_function
from tinybig.data.classic_dataloader import tabular_dataloader, diabetes, iris, banknote, wheat
