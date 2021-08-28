# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
from typing import Dict

import networkx as nx
import numpy as np

from graspologic.partition import modularity, modularity_components
from tests.utils import data_file


def _modularity_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=4.0)
    graph.add_edge("b", "c", weight=3.0)
    graph.add_edge("e", "f", weight=5.0)

    return graph


_PARTITIONS: Dict[str, int] = {"a": 0, "b": 0, "c": 0, "e": 1, "f": 1}


class TestModularity(unittest.TestCase):
    def test_modularity(self):
        graph = _modularity_graph()  # links = 12.0
        partition = _PARTITIONS  # in community degree for -> 0: 14, 1: 10, community degree -> 0:14, 1:10
        # modularity component for partition 0: (14.0 / (2.0 * 12.0)) - (1.0 * ((14.0 / (2 * 12.0)) ** 2.0))
        # (cont): 0.5833333333333334 - 0.34027777777777785 = 0.24305555555555552
        # modularity component for partition 1: (10.0 / (2.0 * 12.0)) - (1.0 * ((10.0 / (2 * 12.0)) ** 2.0))
        # (cont): 0.4166666666666667 - 0.17361111111111113 = 0.24305555555555555
        modularity_value = modularity(graph, partition)

        np.testing.assert_almost_equal(0.48611111111111105, modularity_value)

    def test_modularity_components(self):
        graph = nx.Graph()
        with open(data_file("large-graph.csv"), "r") as edge_list_io:
            for line in edge_list_io:
                source, target, weight = line.strip().split(",")
                previous_weight = graph.get_edge_data(source, target, {"weight": 0})[
                    "weight"
                ]
                weight = float(weight) + previous_weight
                graph.add_edge(source, target, weight=weight)

        partitions = {}
        with open(data_file("large-graph-partitions.csv"), "r") as communities_io:
            for line in communities_io:
                vertex, comm = line.strip().split(",")
                partitions[vertex] = int(comm)

        partition_count = max(partitions.values())

        graph.add_node("disconnected_node")
        partitions["disconnected_node"] = partition_count + 1

        components = modularity_components(graph, partitions)

        # from python-louvain modularity function
        community_modularity = 0.8008595783563607
        total_modularity = sum(components.values())

        self.assertSetEqual(set(components.keys()), set(partitions.values()))
        self.assertEqual(0, components[partition_count + 1])

        np.testing.assert_almost_equal(
            community_modularity, total_modularity, decimal=8
        )
