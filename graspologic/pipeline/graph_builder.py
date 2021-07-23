# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import networkx as nx


__all__ = ["GraphBuilder"]


class GraphBuilder:
    def __init__(self, directed: bool = False):
        """
        GraphBuilder is a simple builder for networkx Graphs. To use less memory,
        it automatically maps all node ids of type ``Any`` to ``int``s.

        The main method it provides, ``add_edge``, will, by default, sum edge weights
        if the edge already exists.

        Parameters
        ----------
        directed : bool
            Default value is ``False``. Used to create either a ``networkx.Graph()`` or
            ``networkx.DiGraph()`` object.
        """
        # OrderedDict is the default for {} anyway, but I wanted to be very explicit,
        # since we absolutely rely on the ordering
        self._id_map = OrderedDict()
        self._graph = nx.DiGraph() if directed else nx.Graph()

    def add_edge(
        self,
        source: Any,
        target: Any,
        weight: float = 1.0,
        sum_weight: bool = True,
        **attributes: Any
    ):
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

    def build(self) -> Tuple[nx.Graph, Dict[Any, int], List[Any]]:
        old_to_new = self._id_map
        new_to_old = [key for key, _ in old_to_new.items()]
        return self._graph, old_to_new, new_to_old

    def _map_node_id(self, node_id: Any) -> int:
        mapped_node_id = self._id_map.get(node_id, len(self._id_map))
        self._id_map[node_id] = mapped_node_id
        return mapped_node_id
