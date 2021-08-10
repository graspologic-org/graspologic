# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numbers
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
from beartype import beartype

__all__ = ["GraphBuilder"]


class GraphBuilder:
    """
    GraphBuilder is a simple builder for networkx Graphs. To use less memory,
    it automatically maps all node ids of any hashable type to ``int``.

    In other words, if you can use it as a key in a dictionary, it will work.

    By default, the main method it provides, ``add_edge``, will sum edge weights
    if the edge already exists.

    Parameters
    ----------
    directed : bool (default=False)
        Used to create either a :class:`networkx.Graph` or
        :class:`networkx.DiGraph` object.
    """

    @beartype
    def __init__(self, directed: bool = False):
        # OrderedDict is the default for {} anyway, but I wanted to be very explicit,
        # since we absolutely rely on the ordering
        self._id_map = OrderedDict()
        self._graph = nx.DiGraph() if directed else nx.Graph()

    @beartype
    def add_edge(
        self,
        source: Any,
        target: Any,
        weight: numbers.Real = 1.0,
        sum_weight: bool = True,
        **attributes: Any
    ) -> None:
        """
        Adds a weighted edge between the provided source and target. The source
        and target id are converted to a unique ``int``.

        If no weight is provided, a default weight of ``1.0`` is used.

        If an edge between the source and target already exists, and if the
        ``sum_weight`` argument is ``True``, then the weights are summed.

        Otherwise, the last weight provided will be used as the edge's weight.

        Any other attributes specified will be added to the edge's data dictionary.

        Parameters
        ----------
        source : Any
            source node id
        target : Any
            target node id
        weight : numbers.Real (default=1.0)
            The weight for the edge. If none is provided, the weight is defaulted to 1.
        sum_weight : bool (default=True)
            If an edge between the ``source`` and ``target`` already exist, should we
            sum the edge weights or overwrite the edge weight with the provided
            ``weight`` value.
        attributes : kwargs
            The attributes kwargs are presumed to be attributes that should be added
            to the edge dictionary for ``source`` and ``target``.
        """
        source_id = self._map_node_id(source)
        target_id = self._map_node_id(target)
        if sum_weight:
            old = self._graph.get_edge_data(source_id, target_id, default={}).get(
                "weight", 0
            )
            self._graph.add_edge(
                source_id, target_id, weight=old + weight, **attributes
            )
        else:
            self._graph.add_edge(source_id, target_id, weight=weight, **attributes)

    def build(self) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[Any, int], List[Any]]:
        """
        Returns
        -------
        Tuple[Union[nx.Graph, nx.DiGraph], Dict[Any, int], List[Any]]
            The returned tuple is either an undirected or directed graph, depending on
            the constructor argument ``directed``. The second value in the tuple is a
            dictionary of original node ids to their assigned integer ids. The third
            and final value in the tuple is a List of original node ids, where the
            index corresponds to the assigned integer and the value is the corresponding
            original ID.
        """
        old_to_new = self._id_map
        new_to_old = [key for key, _ in old_to_new.items()]
        return self._graph, old_to_new, new_to_old

    def _map_node_id(self, node_id: Any) -> int:
        mapped_node_id = self._id_map.get(node_id, len(self._id_map))
        self._id_map[node_id] = mapped_node_id
        return mapped_node_id
