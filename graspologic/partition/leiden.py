# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import networkx as nx

import graspologic_native as gn


def _is_edge_list(graph: Any) -> bool:
    return (
        isinstance(graph, list)
        and len(graph) > 0
        and isinstance(graph[0], tuple)
        and len(graph[0]) == 3
        and isinstance(graph[0][0], str)
        and isinstance(graph[0][1], str)
        and (isinstance(graph[0][2], (int, float)))
    )


def _validate_and_build_edge_list(
    graph: Any, weight_attribute: str
) -> List[Tuple[str, str, float]]:
    if isinstance(graph, nx.Graph) and not graph.is_directed():
        graph = [
            (str(source), str(target), float(weight))
            for source, target, weight in graph.edges(data=weight_attribute)
        ]
    elif not _is_edge_list(graph):
        raise TypeError(
            "graph must be of type List[Tuple[str, str, float]] or be an undirected networkx.Graph"
        )
    return graph


def leiden(
    graph: Union[List[Tuple[str, str, float]], nx.Graph],
    starting_communities: Optional[Dict[str, int]] = None,
    iterations: int = 1,
    resolution: float = 1.0,
    randomness: float = 0.001,
    use_modularity: bool = True,
    random_seed: Optional[int] = None,
    weight_attribute: str = "weight",
) -> Dict[str, int]:
    graph = _validate_and_build_edge_list(graph, weight_attribute)

    improved, modularity, partitions = gn.leiden(
        edges=graph,
        starting_communities=starting_communities,
        resolution=resolution,
        randomness=randomness,
        iterations=iterations,
        use_modularity=use_modularity,
        seed=random_seed,
    )

    return partitions


class HierarchicalCluster(NamedTuple):
    node: str
    cluster: int
    parent_cluster: Optional[int]
    level: int
    is_final_cluster: bool

    @classmethod
    def from_native(
        cls, native_cluster: gn.HierarchicalCluster
    ) -> "HierarchicalCluster":
        return cls(
            node=native_cluster.node,
            cluster=native_cluster.cluster,
            parent_cluster=native_cluster.parent_cluster,
            level=native_cluster.level,
            is_final_cluster=native_cluster.is_final_cluster,
        )


def hierarchical_leiden(
    graph: Union[List[Tuple[str, str, float]], nx.Graph],
    max_cluster_size: int = 1000,
    starting_communities: Optional[Dict[str, int]] = None,
    iterations: int = 1,
    resolution: float = 1.0,
    randomness: float = 0.001,
    use_modularity: bool = True,
    random_seed: Optional[int] = None,
    weight_attribute: str = "weight",
) -> List[HierarchicalCluster]:
    graph = _validate_and_build_edge_list(graph, weight_attribute)
    hierarchical_clusters_native = gn.hierarchical_leiden(
        edges=graph,
        starting_communities=starting_communities,
        resolution=resolution,
        randomness=randomness,
        iterations=iterations,
        use_modularity=use_modularity,
        max_cluster_size=max_cluster_size,
        seed=random_seed,
    )

    return [
        HierarchicalCluster.from_native(entry) for entry in hierarchical_clusters_native
    ]
