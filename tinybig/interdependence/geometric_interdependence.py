# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#############################
# Geometric Interdependence #
#############################

from typing import Union
import torch

from tinybig.interdependence import interdependence
from tinybig.koala.geometry import grid as grid_structure
from tinybig.koala.geometry import cuboid, sphere, cylinder
from tinybig.config.base_config import config


class geometric_interdependence(interdependence):

    def __init__(
        self,
        b: int = 0, m: int = 0,
        interdependence_type: str = 'attribute',
        name: str = 'geometric_interdependence',
        # grid structure initialization options
        grid: grid_structure = None,
        grid_configs: dict = None,
        h: int = None, w: int = None, d: int = 1, channel_num: int = 1,
        # patch structure initialization options
        patch: Union[cuboid, cylinder, sphere] = None,
        patch_configs: dict = None,
        # packing options
        packing_strategy: str = 'densest_packing',
        cd_h: int = None, cd_w: int = None, cd_d: int = None,
        interdependence_matrix_mode: str = 'padding',
        # interdependence matrix processing options
        normalization: bool = False,
        normalization_mode: str = 'row_column',
        # by default,
        require_data: bool = False, require_parameters: bool = False,
        device: str = 'cpu', *args, **kwargs
    ):
        super().__init__(b=b, m=m, name=name, interdependence_type=interdependence_type, require_data=require_data, require_parameters=require_parameters, device=device, *args, **kwargs)

        if grid is not None:
            self.grid = grid
        elif grid_configs is not None:
            self.grid = config.instantiation_from_configs(configs=grid_configs, class_name='grid_class', parameter_name='grid_parameters')
        elif h is not None and w is not None and d is not None:
            grid_parameters = {'h': h, 'w': w, 'd': d, 'universe_num': channel_num}
            self.grid = grid_structure(**grid_parameters)
        else:
            raise ValueError('the grid structure is not specified yet...')

        if interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            if self.m is None or self.m <= 0:
                self.m = self.grid.get_volume(across_universe=True)
            assert self.grid.get_volume(across_universe=True) == self.m
        elif interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            if self.b is None or self.b <= 0:
                self.b = self.grid.get_volume(across_universe=True)
            assert self.grid.get_volume(across_universe=True) == self.b
        else:
            raise ValueError('the interdependence_type is not supported yet...')

        if patch is not None:
            self.patch = patch
        elif patch_configs is not None:
            self.patch = config.instantiation_from_configs(configs=patch_configs, class_name='patch_class', parameter_name='patch_parameters')
        else:
            raise ValueError('the patch structure is not specified yet...')

        self.packing_strategy = packing_strategy
        self.cd_h, self.cd_w, self.cd_d = (cd_h, cd_w, cd_d) if (cd_h is not None and cd_w is not None and cd_d is not None) else self.patch.packing_strategy_parameters(packing_strategy=self.packing_strategy)

        self.interdependence_matrix_mode = interdependence_matrix_mode
        self.normalization = normalization
        self.normalization_mode = normalization_mode

    def update_grid(self, new_grid: grid_structure):
        self.grid = new_grid

    def update_packing_strategy(self, new_packing_strategy: str):
        self.packing_strategy = new_packing_strategy
        self.cd_h, self.cd_w, self.cd_d = self.patch.packing_strategy_parameters(packing_strategy=self.packing_strategy)

    def update_patch(self, new_patch: Union[cuboid, cylinder, sphere]):
        self.patch = new_patch
        self.cd_h, self.cd_w, self.cd_d = self.patch.packing_strategy_parameters(packing_strategy=self.packing_strategy)

    def update_packing_parameters(self, new_cd_h: int, new_cd_w: int, new_cd_d: int):
        self.cd_h = new_cd_h
        self.cd_w = new_cd_w
        self.cd_d = new_cd_d

    def get_channel_num(self):
        return self.grid.get_universe_num()

    def get_patch_size(self):
        return self.patch.get_volume()

    def get_patch_num(self, across_universe: bool = False):
        return self.grid.get_patch_num(cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d, across_universe=across_universe)

    def get_grid_size(self, across_universe: bool = False):
        return self.grid.get_volume(across_universe=across_universe)

    def get_grid_shape(self):
        return self.grid.get_grid_shape()

    def get_grid_shape_after_packing(self):
        return self.grid.get_grid_shape_after_packing(cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d)

    def calculate_b_prime(self, b: int = None, across_universe: bool = True):
        b = b if b is not None else self.b
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            assert self.grid.get_volume(across_universe=across_universe) == b
            if self.interdependence_matrix_mode == 'padding':
                return self.grid.get_patch_num(cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d, across_universe=across_universe) * self.patch.get_volume()
            elif self.interdependence_matrix_mode == 'aggregation':
                return b
        else:
            return b

    def calculate_m_prime(self, m: int = None, across_universe: bool = True):
        m = m if m is not None else self.m
        if self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            assert self.grid.get_volume(across_universe=across_universe) == m
            if self.interdependence_matrix_mode == 'padding':
                return self.grid.get_patch_num(cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d, across_universe=across_universe) * self.patch.get_volume()
            elif self.interdependence_matrix_mode == 'aggregation':
                return m
        else:
            return m

    def calculate_A(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, across_universe: bool = False, device: str = 'cpu', *args, **kwargs):
        if not self.require_data and not self.require_parameters and self.A is not None:
            return self.A
        else:
            A = self.grid.to_matrix(
                patch=self.patch, packing_strategy=self.packing_strategy,
                cd_h=self.cd_h, cd_w=self.cd_w, cd_d=self.cd_d,
                interdependence_matrix_mode=self.interdependence_matrix_mode,
                normalization=self.normalization, normalization_mode=self.normalization_mode,
                across_universe=across_universe, device=device, *args, **kwargs
            )
            A = self.post_process(x=A, device=device)
            if not self.require_data and not self.require_parameters and self.A is None:
                self.A = A
            return A

    def forward(self, x: torch.Tensor = None, w: torch.nn.Parameter = None, kappa_x: torch.Tensor = None, device: str = 'cpu', *args, **kwargs):
        if self.require_data:
            assert x is not None and x.ndim == 2
        if self.require_parameters:
            assert w is not None and w.ndim == 2

        data_x = kappa_x if kappa_x is not None else x
        b, m = data_x.shape
        data_x = data_x.view(b*self.grid.get_universe_num(), -1)
        if self.interdependence_type in ['row', 'left', 'instance', 'instance_interdependence']:
            # A shape: b * b'
            A = self.calculate_A(x.transpose(0, 1), w, device=device)
            assert A is not None and A.size(0) == data_x.size(0)
            if data_x.is_sparse or A.is_sparse:
                xi_x = torch.sparse.mm(A.t(), data_x)
            else:
                xi_x = torch.matmul(A.t(), data_x)
            return xi_x
        elif self.interdependence_type in ['column', 'right', 'attribute', 'attribute_interdependence']:
            # A shape: m * m'
            A = self.calculate_A(x, w, device=device)
            assert A is not None and A.size(0) == data_x.size(-1)
            if data_x.is_sparse or A.is_sparse:
                xi_x = torch.sparse.mm(data_x, A)
            else:
                xi_x = torch.matmul(data_x, A)

            if self.interdependence_matrix_mode == 'padding':
                # shape [b, c, g, p] -> shape [b, g, c, p]
                xi_x = xi_x.view(b, self.grid.get_universe_num(), self.get_patch_num(), self.get_patch_size())
                xi_x = xi_x.permute(0, 2, 1, 3)
            elif self.interdependence_matrix_mode == 'aggregation':
                # shape [b, c, g] -> shape [b, g, c]
                xi_x = xi_x.view(b, self.grid.get_universe_num(), self.get_patch_num())
                xi_x = xi_x.permute(0, 2, 1)
            return xi_x.reshape(b, -1)
        else:
            raise ValueError(f"Invalid interdependence type: {self.interdependence_type}")



class cuboid_patch_based_geometric_interdependence(geometric_interdependence):

    def __init__(
        self,
        p_h: int, p_w: int, p_d: int,
        p_h_prime: int = None, p_w_prime: int = None, p_d_prime: int = None,
        name: str = 'cuboid_patch_based_geometric_interdependence',
        *args, **kwargs
    ):
        patch = cuboid(p_h=p_h, p_w=p_w, p_d=p_d, p_h_prime=p_h_prime, p_w_prime=p_w_prime, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class cuboid_patch_padding_based_geometric_interdependence(cuboid_patch_based_geometric_interdependence):
    def __init__(self, interdependence_matrix_mode: str = 'padding', *args, **kwargs):
        super().__init__(interdependence_matrix_mode='padding', *args, **kwargs)


class cuboid_patch_aggregation_based_geometric_interdependence(cuboid_patch_based_geometric_interdependence):
    def __init__(self, interdependence_matrix_mode: str = 'aggregation', *args, **kwargs):
        super().__init__(interdependence_matrix_mode='aggregation', *args, **kwargs)


class cylinder_patch_based_geometric_interdependence(geometric_interdependence):

    def __init__(
        self,
        r: int, p_d: int, p_d_prime: int = None,
        name: str = 'cylinder_patch_based_geometric_interdependence',
        *args, **kwargs
    ):

        patch = cylinder(r=r, p_d=p_d, p_d_prime=p_d_prime)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class cylinder_patch_padding_based_geometric_interdependence(cylinder_patch_based_geometric_interdependence):
    def __init__(self, interdependence_matrix_mode: str = 'padding', *args, **kwargs):
        super().__init__(interdependence_matrix_mode='padding', *args, **kwargs)


class cylinder_patch_aggregation_based_geometric_interdependence(cylinder_patch_based_geometric_interdependence):
    def __init__(self, interdependence_matrix_mode: str = 'aggregation', *args, **kwargs):
        super().__init__(interdependence_matrix_mode='aggregation', *args, **kwargs)


class sphere_patch_based_geometric_interdependence(geometric_interdependence):

    def __init__(
        self,
        r: int,
        name: str = 'sphere_patch_based_geometric_interdependence',
        *args, **kwargs
    ):

        patch = sphere(r=r)
        super().__init__(name=name, patch=patch, *args, **kwargs)


class sphere_patch_padding_based_geometric_interdependence(sphere_patch_based_geometric_interdependence):
    def __init__(self, interdependence_matrix_mode: str = 'padding', *args, **kwargs):
        super().__init__(interdependence_matrix_mode='padding', *args, **kwargs)


class sphere_patch_aggregation_based_geometric_interdependence(sphere_patch_based_geometric_interdependence):
    def __init__(self, interdependence_matrix_mode: str = 'aggregation', *args, **kwargs):
        super().__init__(interdependence_matrix_mode='aggregation', *args, **kwargs)


if __name__ == '__main__':
    import torch
    h = 4
    w = 5
    d = 3
    img = torch.randint(low=0, high=10, size=(d, h, w), dtype=torch.float32, device='mps')
    x = img.flatten()

    print(img, x)
    grid = grid_structure(h=h, w=w, d=d)
    patch = cuboid(p_h=1, p_w=1, p_d=0, p_d_prime=d)
    print(patch.get_volume())
    interdependence_func = geometric_interdependence(
        grid=grid, patch=patch, cd_h=1, cd_w=1, cd_d=1,
        interdependence_matrix_mode='padding'
    )
    A = interdependence_func(device='mps')
    np_array = A.cpu().numpy()

    xi_x = torch.matmul(x, A)
    for row in xi_x.view(-1, patch.get_volume()):
        print("[" + " ".join(f"{val:.2f}" for val in row) + "]")

    for row in np_array:
        print("[" + " ".join(f"{val:.2f}" for val in row) + "]")

    print(torch.sum(A, dim=0))


