# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Chain Topological Structure #
###############################

from tinybig.koala.topology import base_topology


class chain(base_topology):
    """
    Represents a chain topology, a linear sequence of connected nodes.

    Parameters
    ----------
    length : int
        The number of nodes in the chain. Must be a positive integer.
    name : str, optional
        The name of the topology, by default 'chain'.
    bi_directional : bool, optional
        Specifies whether the chain is bidirectional. If True, links between nodes are bidirectional.
        By default, False (unidirectional).
    *args : tuple
        Additional positional arguments to be passed to the `base_topology` class.
    **kwargs : dict
        Additional keyword arguments to be passed to the `base_topology` class.

    Raises
    ------
    ValueError
        If `length` is not provided or is less than 1.

    Inherits From
    -------------
    base_topology
        The base class for topological structures.
    """
    def __init__(self, length: int, name: str = 'chain', bi_directional: bool = False, *args, **kwargs):
        """
        Initializes the chain topology.

        Parameters
        ----------
        length : int
            The number of nodes in the chain. Must be a positive integer.
        name : str, optional
            The name of the topology, by default 'chain'.
        bi_directional : bool, optional
            Specifies whether the chain is bidirectional. If True, links between nodes are bidirectional.
            By default, False (unidirectional).
        *args : tuple
            Additional positional arguments to be passed to the `base_topology` class.
        **kwargs : dict
            Additional keyword arguments to be passed to the `base_topology` class.

        Raises
        ------
        ValueError
            If `length` is not provided or is less than 1.
        """
        if length is None or length < 1:
            raise ValueError('A positive length needs to be specified')
        nodes = list(range(length))
        links = [(i, i + 1) for i in range(length-1)]
        super().__init__(name=name, nodes=nodes, links=links, directed=not bi_directional, *args, **kwargs)

    def length(self):
        """
        Returns the number of links in the chain, which corresponds to the size of the topology.

        Returns
        -------
        int
            The number of links in the chain.
        """
        return self.size()




