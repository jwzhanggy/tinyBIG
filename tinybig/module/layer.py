# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch
import torch.nn.functional as F
import warnings

from tinybig.module import rpn_head
from tinybig.util import get_obj_from_str, process_num_alloc_configs


class rpn_layer(torch.nn.Module):

    def __init__(self, m: int, n: int, head_num: int = None, fusion_strategy: str = 'average',
                 heads: list = None, head_num_alloc: int | list = None,
                 head_configs: dict | list = None, device='cpu', *args, **kwargs):
        super().__init__()
        assert m is not None and n is not None
        self.m = m
        self.n = n
        self.fusion_strategy = fusion_strategy
        self.fusion_parameters = None
        self.heads = torch.nn.ModuleList()

        if heads is not None:
            # initialize heads from the provided head parameter directly
            self.heads.extend(heads)
        elif head_configs is None:
            raise ValueError("Both heads and head_configs are None, the layer cannot be initialized...")
        else:
            # initialize heads from the head configs
            head_num, head_num_alloc, head_configs = process_num_alloc_configs(head_num, head_num_alloc, head_configs)
            assert len(head_num_alloc) == len(head_configs) and sum(head_num_alloc) == head_num

            self.head_num = head_num
            self.head_num_alloc = head_num_alloc
            self.head_configs = head_configs
            for head_repeat_time, head_config in zip(self.head_num_alloc, self.head_configs):
                for head_id in range(0, head_repeat_time):
                    head_class_name = head_config['head_class']
                    head_parameters = head_config['head_parameters']
                    head_parameters['m'] = self.m
                    head_parameters['n'] = self.n
                    head_parameters['device'] = device
                    self.heads.append(get_obj_from_str(head_class_name)(**head_parameters))

        assert len(self.heads) == self.head_num and [(self.m, self.n)] * self.head_num == [(head.m, head.n) for head in self.heads]
        # make self.head_num_alloc consistent with head in self.heads
        self.head_num_alloc = [1] * self.head_num

    def add_single_head(self, index: int, head: rpn_head = None, head_config: dict = None):
        if head is None and head_config is None:
            raise ValueError("Neither head or head_config is provided, please provide the information of the head to be added...")
        head = head if head is not None else rpn_head(*head_config)
        self.head_num_alloc.insert(index, 1)
        self.heads.insert(index, head)
        self.head_num += 1
        assert len(self.heads) == self.head_num

    def add_heads(self, index: int | list = -1, repeat_times: int = None,
                  head: rpn_head | list = None, head_config: dict | list = None,
                  *args, **kwargs):
        index_list = None
        if type(index) is int:
            if repeat_times is not None:
                index_list = [index] * repeat_times
        else:
            index_list = index

        if head is not None:
            if type(head) is rpn_head:
                head_list = [head] * index_list
            else:
                head_list = head
        elif head_config is not None:
            if type(head_config) is dict:
                head_list = [rpn_head(head_config=head_config)] * index_list
            else:
                head_list = [rpn_head(config) for config in head_config]
        else:
            raise ValueError("Neither head or head_config is provided, please provide the information of the head to be added...")

        assert len(index_list) == len(head_list)
        for index, head in zip(index_list, head_list):
            self.add_single_head(index=index, head=head)

    def delete_single_head(self, index: int):
        assert index in range(self.head_num)
        self.heads.pop(index)
        self.head_num_alloc.pop(index)
        self.head_num -= 1
        assert len(self.heads) == self.head_num

    def delete_heads(self, index: int | list):
        if type(index) is int:
            self.delete_single_head(index=index)
        else:
            for id in index:
                self.delete_single_head(index=id)

    def initialize_parameters(self):
        for head in self.heads:
            head.initialize_parameters()

    def initialize_fusion_parameters(self):
        self.fusion_parameters = torch.nn.Parameter(torch.rand(self.n, self.n*self.head_num))
        torch.nn.init.xavier_uniform_(self.fusion_parameters)

    def __call__(self, x: torch.Tensor, fusion_strategy=None, device='cpu', *args, **kwargs):
        return self.forward(x=x, fusion_strategy=fusion_strategy, device=device, *args, **kwargs)

    def forward(self, x: torch.Tensor, fusion_strategy=None, device='cpu', *args, **kwargs):
        results = []
        for head in self.heads:
            results.append(head(x=x, device=device))
        # make sure all head results have consistent output shapes prior to fusion
        assert results != [] and [results[0].shape] * len(results) == [result.shape for result in results]
        results = torch.stack(results, dim=0)
        fusion_strategy = fusion_strategy if fusion_strategy is not None else self.fusion_strategy
        if fusion_strategy in ['average', 'mean']:
            return torch.mean(torch.Tensor(results), dim=0)
        elif fusion_strategy in ['sum', 'add']:
            return torch.sum(torch.Tensor(results), dim=0)
        elif fusion_strategy in ['linear']:
            if self.fusion_parameters is None:
                self.initialize_fusion_parameters()
            return F.linear(torch.Tensor(results), self.fusion_parameters)
        else:
            raise ValueError("The fusion strategy {} is not recognized...".format(fusion_strategy))

