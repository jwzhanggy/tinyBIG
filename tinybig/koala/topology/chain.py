# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Chain Topological Structure #
###############################

from tinybig.koala.topology import base_topology


class chain(base_topology):

    def __init__(self, length: int, name: str = 'chain', bi_directional: bool = False, *args, **kwargs):
        if length is None or length < 1:
            raise ValueError('A positive length needs to be specified')
        nodes = list(range(length))
        links = [(i, i + 1) for i in range(length-1)]
        super().__init__(name=name, nodes=nodes, links=links, directed=not bi_directional, *args, **kwargs)

    def length(self):
        return self.size()




