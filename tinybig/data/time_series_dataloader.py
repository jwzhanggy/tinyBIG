# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Time Series Dataloader #
##########################

import warnings
import numpy as np
from typing import Union

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from tinybig.data import dataloader, dataset
from tinybig.util import check_file_existence, download_file_from_github, create_directory_if_not_exists, unzip_file


class time_series_dataloader(dataloader):
    def __init__(
        self,
        data_profile: dict,
        x_len: int, y_len: int,
        xy_gap: int = 1,
        name: str = 'time_series_dataloader',
        time_granularity: str = 'daily',
        target_attributes: str = 'All',
        coverage_year_range: int = 1,
        instance_id: Union[int, str] = None,
        train_batch_size: int = 64,
        test_batch_size: int = 64,
    ):
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
        self.data_profile = data_profile

        self.x_len = x_len
        self.y_len = y_len
        self.xy_gap = xy_gap
        self.time_granularity = time_granularity
        self.target_attributes = target_attributes
        self.coverage_year_range = coverage_year_range
        self.instance_id = instance_id

    def get_data_profile(self):
        return self.data_profile

    def get_data_name(self):
        assert self.data_profile is not None
        return self.data_profile['name']

    def download_data(self, cache_dir: str, file_name: str, time_granularity: str):
        if cache_dir is None or file_name is None or time_granularity is None:
            raise ValueError("The cache directory, file name and time_granularity must be specified.")

        if 'zipped_files' in self.data_profile and file_name in self.data_profile['zipped_files']:
            postfix = '.zip'
        else:
            postfix = ''

        complete_file_path = f'{cache_dir}/{time_granularity}/{file_name}{postfix}'
        url = f'{self.data_profile['url']['url_prefix']}/{time_granularity}/{file_name}{postfix}'
        create_directory_if_not_exists(complete_file_path=complete_file_path)
        download_file_from_github(url_link=url, destination_path=complete_file_path)

        if postfix == '.zip':
            unzip_file(complete_file_path=complete_file_path)

    def load_raw(self, cache_dir: str, file_name: str,  time_granularity: str, device: str = 'cpu'):
        if cache_dir is None or file_name is None or time_granularity is None:
            raise ValueError("The cache directory, file name and time_granularity must be specified.")

        if not check_file_existence(f'{cache_dir}/{time_granularity}/{file_name}'):
            self.download_data(cache_dir=cache_dir, file_name=file_name, time_granularity=time_granularity)

        data = np.loadtxt(f'{cache_dir}/{time_granularity}/{file_name}', delimiter=',', dtype='str')
        instance_ids = data[0, 1:]
        timestamps = data[1:, 0]
        time_series_data = data[1:, 1:].astype(float)
        time_series_data = torch.tensor(time_series_data, dtype=torch.float).to(device)

        return instance_ids.tolist(), timestamps.tolist(), time_series_data

    def partition_data(self, data_batch: torch.Tensor, x_len: int, y_len: int, xy_gap: int):
        t, n = data_batch.shape

        if t < x_len + y_len + xy_gap:
            raise ValueError("The data batch size must be larger than the number of data points.")

        X, Y = [], []
        for start_idx in range(0, t - x_len - y_len - xy_gap + 1):
            x_segment = data_batch[start_idx:start_idx+x_len, :]
            y_segment = data_batch[start_idx+x_len+xy_gap:start_idx+x_len+xy_gap+y_len, :]
            X.append(x_segment)
            Y.append(y_segment)
        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def load(
        self,
        # directory to load the data
        cache_dir: str = None,
        # parameters to locate files
        time_granularity: str = None,
        target_attributes: str = None,
        coverage_year_range: int = None,
        # data partition parameters
        instance_id: Union[int, str] = None,
        train_percentage: float = 0.8,
        # other parameters
        device: str = 'cpu',
        *args, **kwargs
    ):
        cache_dir = cache_dir if cache_dir is not None else f'./data/{self.data_profile['name']}'
        target_attributes = target_attributes if target_attributes is not None else self.target_attributes
        time_granularity = time_granularity if time_granularity is not None else self.time_granularity
        instance_id = instance_id if instance_id is not None else self.instance_id

        assert target_attributes in self.data_profile['target_attributes']
        assert time_granularity in self.data_profile['time_granularity']

        if 'coverage_year_range' in self.data_profile:
            coverage_year_range = coverage_year_range if coverage_year_range is not None else self.coverage_year_range
            assert coverage_year_range in self.data_profile['coverage_year_range']
            file_name = f'{coverage_year_range}_year_{time_granularity}_{target_attributes}.csv'
        else:
            file_name = f'{time_granularity}_{target_attributes}.csv'

        instance_ids, timestamps, time_series_data = self.load_raw(cache_dir=cache_dir, time_granularity=time_granularity, file_name=file_name)

        if instance_id is not None:
            if instance_id in instance_ids:
                column_index = instance_ids.index(instance_id)
                time_series_data = time_series_data[:, column_index:column_index+1]
            else:
                warnings.warn("Unrecognized data id " + str(instance_id) + ' ...')

        X, y = self.partition_data(data_batch=time_series_data, x_len=self.x_len, y_len=self.y_len, xy_gap=self.xy_gap)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=int(train_percentage * len(X)),
            shuffle=False
        )

        train_dataset = dataset(X_train, y_train)
        test_dataset = dataset(X_test, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)
        return {'train_loader': train_loader, 'test_loader': test_loader}


ETF_DATA_PROFILE = {
    'name': 'etfs',
    'time_granularity': ('daily', 'weekly', 'monthly', 'quarterly', 'yearly'),
    'target_attributes': ('All', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt'),
    'coverage_year_range': (1, 3, 5, 10),
    'url': {
        'url_prefix': 'https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/time_series/etfs/'
    }
}

STOCK_DATA_PROFILE = {
    'name': 'stocks',
    'time_granularity': ('daily', 'weekly', 'monthly', 'quarterly', 'yearly'),
    'target_attributes': ('All', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt'),
    'coverage_year_range': (1, 3, 5, 10, 20, 30),
    'url': {
        'url_prefix': 'https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/time_series/stocks/'
    },
    'zipped_files': [
        '3_year_daily_All.csv',
        '5_year_daily_All.csv',
        '10_year_daily_All.csv',
    ]
}

TRAFFIC_LA_DATA_PROFILE = {
    'name': 'traffic_la',
    'time_granularity': ('minutely', '6hourly', '12hourly', 'hourly', 'daily', 'weekly', 'monthly'),
    'target_attributes': ('All',),
    'url': {
        'url_prefix': 'https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/time_series/traffic_la/'
    }
}

TRAFFIC_BAY_DATA_PROFILE = {
    'name': 'traffic_bay',
    'time_granularity': ('minutely', '6hourly', '12hourly', 'hourly', 'daily', 'weekly', 'monthly'),
    'target_attributes': ('All',),
    'url': {
        'url_prefix': 'https://raw.githubusercontent.com/jwzhanggy/tinybig_dataset_repo/main/data/time_series/traffic_bay/'
    }
}


class etf(time_series_dataloader):
    def __init__(self, name: str = 'etf', *args, **kwargs):
        super().__init__(data_profile=ETF_DATA_PROFILE, name=name, *args, **kwargs)


class stock(time_series_dataloader):
    def __init__(self, name: str = 'stock', *args, **kwargs):
        super().__init__(data_profile=STOCK_DATA_PROFILE, name=name, *args, **kwargs)


class traffic_la(time_series_dataloader):
    def __init__(self, name: str = 'traffic_la', *args, **kwargs):
        super().__init__(data_profile=TRAFFIC_LA_DATA_PROFILE, name=name, *args, **kwargs)


class traffic_bay(time_series_dataloader):
    def __init__(self, name: str = 'traffic_bay', *args, **kwargs):
        super().__init__(data_profile=TRAFFIC_BAY_DATA_PROFILE, name=name, *args, **kwargs)


if __name__ == '__main__':
    data_loader = time_series_dataloader(data_profile=STOCK_DATA_PROFILE, train_len=32, test_len=1)
    cache_dir = './data/stock'
    time_granularity = 'daily'
    file_name = '10_year_daily_All.csv'
    instance_ids, timestamps, time_series_data = data_loader.load_raw(cache_dir=cache_dir, file_name=file_name, time_granularity=time_granularity)
    print(instance_ids)
    print(timestamps)
    print(time_series_data.shape)

