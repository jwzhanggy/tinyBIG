# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis


import pickle
from abc import abstractmethod
from tinybig.util.util import create_directory_if_not_exists


class result:
    def __init__(self, name='base_result', *args, **kwargs):
        self.name = name

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass


class prediction(result):

    def __init__(self, name='prediction_result'):
        super().__init__(name=name)

    def save(self, results, cache_dir='./result', result_file='prediction_result'):
        create_directory_if_not_exists(f"{cache_dir}/{result_file}")
        with open(f"{cache_dir}/{result_file}", 'wb') as f:
            pickle.dump(results, f)

    def load(self, cache_dir='./result', result_file='prediction_result'):
        with open("{}/{}".format(cache_dir, result_file), 'rb') as f:
            results = pickle.load(f)
        return results

