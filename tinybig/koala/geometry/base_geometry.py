# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

#################
# Base Geometry #
#################

class base_geometry:

    def __init__(self, name: str='base_geometric_structure', *args, **kwargs):
        self.name = name

    def get_name(self) -> str:
        return self.name