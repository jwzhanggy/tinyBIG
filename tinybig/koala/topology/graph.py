# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Graph Topological Structure #
###############################

from tinybig.koala.topology import base_topology


class graph(base_topology):

    def __init__(self, name: str = 'graph', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def bfs(self, start=None, goal=None):
        pass

    def dfs(self, start=None, goal=None):
        pass


