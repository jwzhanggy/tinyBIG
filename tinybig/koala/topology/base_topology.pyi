# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################
# Base Topology #
#################

from abc import abstractmethod


class base_topology:

    def __init__(self, name: str='base_topological_structure', *args, **kwargs):
        self.name = name

    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def to_matrix(self, *args, **kwargs):
        pass

