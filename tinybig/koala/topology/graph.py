# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

###############################
# Graph Topological Structure #
###############################

from tinybig.koala.topology import base_topology


class graph(base_topology):
    """
    Represents a generic graph structure, inheriting from the `base_topology` class.

    Parameters
    ----------
    name : str, optional
        The name of the graph, by default 'graph'.
    *args : tuple
        Additional positional arguments to be passed to the `base_topology` class.
    **kwargs : dict
        Additional keyword arguments to be passed to the `base_topology` class.

    Inherits From
    -------------
    base_topology
        The base class for topological structures.
    """
    def __init__(self, name: str = 'graph', *args, **kwargs):
        """
        Initializes a graph with the specified name.

        Parameters
        ----------
        name : str, optional
            The name of the graph, by default 'graph'.
        *args : tuple
            Additional positional arguments to be passed to the `base_topology` class.
        **kwargs : dict
            Additional keyword arguments to be passed to the `base_topology` class.
        """
        super().__init__(name=name, *args, **kwargs)

    def bfs(self, start=None, goal=None):
        """
        Performs a Breadth-First Search (BFS) traversal of the graph.

        Parameters
        ----------
        start : node, optional
            The starting node for the BFS traversal. If None, traversal starts from an arbitrary node.
        goal : node, optional
            The target node to search for. If None, the BFS traversal continues until all nodes are visited.

        Returns
        -------
        list
            A list of nodes representing the BFS traversal path or the path to the goal if specified.
        """
        pass

    def dfs(self, start=None, goal=None):
        """
        Performs a Depth-First Search (DFS) traversal of the graph.

        Parameters
        ----------
        start : node, optional
            The starting node for the DFS traversal. If None, traversal starts from an arbitrary node.
        goal : node, optional
            The target node to search for. If None, the DFS traversal continues until all nodes are visited.

        Returns
        -------
        list
            A list of nodes representing the DFS traversal path or the path to the goal if specified.
        """
        pass

    def radius(self):
        """
        Calculates the radius of the graph.

        Returns
        -------
        int
            The radius of the graph, defined as the minimum eccentricity of all nodes.

        Notes
        -----
        Eccentricity is the maximum distance from a node to any other node in the graph.
        """
        pass

    def shortest_path(self, start=None, goal=None):
        """
        Finds the shortest path between two nodes in the graph.

        Parameters
        ----------
        start : node, optional
            The starting node for the path. If None, raises a ValueError.
        goal : node, optional
            The target node for the path. If None, raises a ValueError.

        Returns
        -------
        list
            A list of nodes representing the shortest path from the start node to the goal node.

        Raises
        ------
        ValueError
            If either the `start` or `goal` node is None or not in the graph.
        """
        pass


