# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Time Series Dataloader #
##########################

from typing import Union

from tinybig.data.base_data import dataloader, dataset


class text_dataloader(dataloader):
    def __init__(self, train_length: Union[int, str] = '1M', test_length: Union[int, str] = '1D', train_test_gap: Union[int, str] = '1D', instance_id: Union[int, str] = None, time_window: str = '3Y', granularity: str = 'D', train_batch_size: int = 64, test_batch_size: int = 64, name: str = 'text_dataloader'):
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)

    def group_with_granularity(self, granularity: str):
        pass

    def load_raw(self):
        pass

    def load(self):
        pass