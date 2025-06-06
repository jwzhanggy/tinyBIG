# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

##########################
# Spatial Entity Classes #
##########################

import torch


class space:
    def __init__(
        self,
        name: str = 'space',
        dim: int = 3,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        *args, **kwargs
    ):
        self.name = name
        if dim not in [2, 3]:
            raise ValueError('dim must be 2 or 3')
        self.dim = dim

        self.device = device
        self.dtype = dtype

    def get_name(self):
        return self.name

    def change_name(self, new_name: str):
        self.name = new_name

    @property
    def dim(self) -> int:
        """
        Returns the spatial dimension of the base object (2 or 3).

        Returns
        -------
        int
            Dimension (2 or 3).
        """
        return self.dim

    @dim.setter
    def dim(self, new_dim: int):
        if new_dim not in [2, 3]:
            raise ValueError('new_dim must be 2 or 3')
        self.dim = new_dim

    def change_device(self, new_device: str):
        self.device = new_device

    def change_dtype(self, new_dtype: torch.dtype):
        self.dtype = new_dtype

