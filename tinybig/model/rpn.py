# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

import torch

from tinybig.util.util import get_obj_from_str, process_num_alloc_configs, create_directory_if_not_exists
from tinybig.module.layer import rpn_layer
from tinybig.model.model import model


class rpn(torch.nn.Module, model):
    def __init__(self, name='Reconciled_Polynomial_Network',
                 depth: int = None, layer_num_alloc: int | list = None,
                 layers: list = None, layer_configs: dict | list = None,
                 device='cpu', *args, **kwargs):
        torch.nn.Module.__init__(self)
        model.__init__(self, name=name, *args, **kwargs)
        self.name = name
        self.layers = torch.nn.ModuleList()

        if layers is not None:
            self.layers.extend(layers)
        elif layer_configs is None:
            raise ValueError("Both layers and layer_configs are None, the model cannot be initialized...")
        else:
            # initialize layers from the layer configs
            depth, layer_num_alloc, layer_configs = process_num_alloc_configs(depth, layer_num_alloc, layer_configs)
            assert len(layer_num_alloc) == len(layer_configs) and depth == sum(layer_num_alloc)

            self.depth = depth
            self.layer_num_alloc = layer_num_alloc
            self.layer_configs = layer_configs
            for layer_repeat_time, layer_config in zip(layer_num_alloc, layer_configs):
                for layer_id in range(layer_repeat_time):
                    class_name = layer_config['layer_class']
                    parameters = layer_config['layer_parameters'] if 'layer_parameters' in layer_config else {}
                    parameters['device'] = device
                    layer = get_obj_from_str(class_name)(**parameters)
                    self.layers.append(layer)

        assert len(self.layers) == self.depth
        self.layer_num_alloc = [1] * self.depth
        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.layers:
            layer.initialize_parameters()

    def add_single_layer(self, index: int, layer: rpn_layer = None, layer_config: dict = None):
        if layer is None and layer_config is None:
            raise ValueError("Neither layer or layer_config is provided, please provide the information of the layer to be added...")
        layer = layer if layer is not None else rpn_layer(layer_config)
        self.layer_num_alloc.insert(index, 1)
        self.layers.insert(index, layer)
        self.depth += 1
        assert len(self.layers) == self.depth

    def add_layers(self, index: int | list = -1, repeat_times: int = None,
                  layer: rpn_layer | list = None, layer_config: dict | list = None,
                  *args, **kwargs):
        index_list = []
        if type(index) is int:
            if repeat_times is not None:
                index_list = [index] * repeat_times
        else:
            index_list = index

        if layer is not None:
            if type(layer) is rpn_layer:
                layer_list = [layer] * index_list
            else:
                layer_list = layer
        elif layer_config is not None:
            if type(layer_config) is dict:
                layer_list = [rpn_layer(layer_config=layer_config)] * index_list
            else:
                layer_list = [rpn_layer(config) for config in layer_config]
        else:
            raise ValueError("Neither layer or layer_config is provided, please provide the information of the layer to be added...")

        assert len(index_list) == len(layer_list)
        for index, layer in zip(index_list, layer_list):
            self.add_single_head(index=index, layer=layer)

        assert len(self.layers) == self.depth
        return True

    def delete_single_layer(self, index: int):
        assert index in range(self.depth)
        self.layers.pop(index)
        self.layer_num_alloc.pop(index)
        self.depth -= 1
        assert len(self.layers) == self.depth

    def delete_layers(self, index: int | list):
        if type(index) is int:
            self.delete_single_layer(index=index)
        else:
            for id in index:
                self.delete_single_layer(index=id)
        return True

    def forward(self, x: torch.Tensor, device='cpu', *args, **kwargs):
        for layer in self.layers:
            x = layer(x, device=device)
        return x

    def save_ckpt(self, cache_dir='./ckpt', checkpoint_file='checkpoint'):
        create_directory_if_not_exists(f'{cache_dir}/{checkpoint_file}')
        torch.save(self.state_dict(), f'{cache_dir}/{checkpoint_file}')

    def load_ckpt(self, cache_dir='./ckpt', checkpoint_file='checkpoint'):
        self.load_state_dict(torch.load(f'{cache_dir}/{checkpoint_file}'), strict=True)
