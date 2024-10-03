# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##################################
# Geometric Compression Function #
##################################

from typing import Union, Callable
import torch

from tinybig.compression import transformation
from tinybig.koala.linear_algebra import batch_max, batch_min
from tinybig.koala.statistics import batch_mean
from tinybig.koala.geometry import grid as grid_structure
from tinybig.koala.geometry import cuboid, sphere, cylinder
from tinybig.interdependence.geometric_interdependence import geometric_interdependence


class geometric_compression(transformation):

    def __init__(
        self,
        metric: Callable[[torch.Tensor], torch.Tensor],
        name: str = 'geometric_compression',
        grid: grid_structure = None,
        grid_configs: dict = None,
        h: int = None, w: int = None, d: int = None,
        patch: Union[cuboid, cylinder, sphere] = None,
        patch_configs: dict = None,
        packing_strategy: str = None,
        cd_h: int = None, cd_w: int = None, cd_d: int = None,
        *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)

        self.metric = metric

        self.interdependence = geometric_interdependence(
            b=0, m=0, interdependence_type='attribute', interdependence_matrix_mode='padding',
            grid=grid, grid_configs=grid_configs, h=h, w=w, d=d,
            patch=patch, patch_configs=patch_configs,
            packing_strategy=packing_strategy, cd_h=cd_h, cd_w=cd_w, cd_d=cd_d,
            normalization=False, require_data=False, require_parameters=False,
        )

    def get_grid_shape(self):
        return self.interdependence.get_grid_shape()

    def get_grid_size(self, across_universe: bool = False):
        return self.interdependence.get_grid_size(across_universe=across_universe)

    def get_patch_size(self):
        return self.interdependence.get_patch_size()

    def get_patch_num(self, across_universe: bool = False):
        return self.interdependence.get_patch_num(across_universe=across_universe)

    def get_grid_shape_after_packing(self):
        return self.interdependence.get_grid_shape_after_packing()

    def calculate_D(self, m: int):
        assert m == self.get_grid_size(across_universe=True)
        return self.interdependence.get_patch_num(across_universe=True)

    def forward(self, x: torch.Tensor, device: str = 'cpu', *args, **kwargs) -> torch.Tensor:
        b, m = x.shape
        x = self.pre_process(x=x, device=device)

        x = self.interdependence(x, device=device)

        p = self.get_patch_size()
        compression = self.metric(x.view(-1, p), dim=1).view(b, self.get_patch_num(), -1)
        compression = compression.permute(0, 2, 1).reshape(b, -1)

        assert compression.shape == (b, self.calculate_D(m=m))
        return self.post_process(x=compression, device=device)


class cuboid_patch_based_geometric_compression(geometric_compression):

    def __init__(
        self,
        p_h: int, p_w: int, p_d: int,
        p_h_prime: int = None, p_w_prime: int = None, p_d_prime: int = None,
        name: str = 'cuboid_patch_based_geometric_compression',
        *args, **kwargs
    ):
        patch = cuboid(p_h=p_h, p_w=p_w, p_d=p_d, p_h_prime=p_h_prime, p_w_prime=p_w_prime, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class cylinder_patch_based_geometric_compression(geometric_compression):

    def __init__(
        self,
        p_r: int, p_d: int, p_d_prime: int = None,
        name: str = 'cylinder_patch_based_geometric_compression',
        *args, **kwargs
    ):
        patch = cylinder(p_r=p_r, p_d=p_d, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class sphere_patch_based_geometric_compression(geometric_compression):

    def __init__(
        self,
        p_r: int,
        name: str = 'sphere_patch_based_geometric_compression',
        *args, **kwargs
    ):

        patch = sphere(p_r=p_r)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class cuboid_max_based_geometric_compression(cuboid_patch_based_geometric_compression):
    def __init__(self, name: str = 'cuboid_max_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class cuboid_min_based_geometric_compression(cuboid_patch_based_geometric_compression):
    def __init__(self, name: str = 'cuboid_min_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class cuboid_mean_based_geometric_compression(cuboid_patch_based_geometric_compression):
    def __init__(self, name: str = 'cuboid_mean_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)


class cylinder_max_based_geometric_compression(cylinder_patch_based_geometric_compression):
    def __init__(self, name: str = 'cylinder_max_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class cylinder_min_based_geometric_compression(cylinder_patch_based_geometric_compression):
    def __init__(self, name: str = 'cylinder_min_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class cylinder_mean_based_geometric_compression(cylinder_patch_based_geometric_compression):
    def __init__(self, name: str = 'cylinder_mean_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)


class sphere_max_based_geometric_compression(sphere_patch_based_geometric_compression):
    def __init__(self, name: str = 'sphere_max_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_max, *args, **kwargs)


class sphere_min_based_geometric_compression(sphere_patch_based_geometric_compression):
    def __init__(self, name: str = 'sphere_min_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_min, *args, **kwargs)


class sphere_mean_based_geometric_compression(sphere_patch_based_geometric_compression):
    def __init__(self, name: str = 'sphere_mean_based_geometric_compression', metric: Callable[[torch.Tensor], torch.Tensor] = None, *args, **kwargs):
        super().__init__(name=name, metric=batch_mean, *args, **kwargs)
