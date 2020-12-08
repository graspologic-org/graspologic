# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

import unittest
import networkx as nx

from graspologic.partition import HierarchicalCluster, hierarchical_leiden, leiden

from tests.utils import data_file


class TestHierarchicalCluster(unittest.TestCase):
    def test_from_native(self):
        with self.assertRaises(TypeError):
            HierarchicalCluster.from_native("foo")

        # note: it is impossible to create a native instance of a HierarchicalCluster.  We will
        # test from_native indirectly through calling graspologic.partition.hierarchical_leiden()

    def test_final_hierarchical_clustering(self):
        with self.assertRaises(TypeError):
            HierarchicalCluster.final_hierarchical_clustering("I am not a list")

        hierarchical_clusters = [
            HierarchicalCluster("1", 0, None, 0, False),
            HierarchicalCluster("2", 0, None, 0, False),
            HierarchicalCluster("3", 0, None, 0, False),
            HierarchicalCluster("4", 1, None, 0, True),
            HierarchicalCluster("5", 1, None, 0, True),
            HierarchicalCluster("1", 2, 0, 1, True),
            HierarchicalCluster("2", 2, 0, 1, True),
            HierarchicalCluster("3", 3, 0, 1, True),
        ]

        expected = {
            "1": 2,
            "2": 2,
            "3": 3,
            "4": 1,
            "5": 1,
        }
        self.assertEqual(
            expected,
            HierarchicalCluster.final_hierarchical_clustering(hierarchical_clusters),
        )


def _create_edge_list() -> List[Tuple[str, str, float]]:
    edges = []
    with open(data_file("large-graph.csv"), "r") as edges_io:
        for line in edges_io:
            source, target, weight = line.strip().split(",")
            edges.append((source, target, float(weight)))
    return edges


class TestLeiden(unittest.TestCase):
    def test_correct_types(self):
        # both leiden and hierarchical_leiden require the same types and mostly the same value range restrictions
        good_args = {
            "starting_communities": {"1": 2},
            "extra_forced_iterations": 0,
            "resolution": 1.0,
            "randomness": 0.001,
            "use_modularity": True,
            "random_seed": None,
            "is_weighted": True,
            "weight_default": 1.0,
            "check_directed": True,
        }

        graph = nx.Graph()
        graph.add_edge("1", "2", weight=3.0)
        graph.add_edge("2", "3", weight=4.0)

        leiden(graph=graph, **good_args)
        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["starting_communities"] = 123
            leiden(graph=graph, **args)

        args = good_args.copy()
        args["starting_communities"] = None
        leiden(graph=graph, **args)

        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["extra_forced_iterations"] = 1234.003
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["extra_forced_iterations"] = -4003
            leiden(graph=graph, **args)

        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["resolution"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["resolution"] = 0
            leiden(graph=graph, **args)

        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["randomness"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["randomness"] = 0
            leiden(graph=graph, **args)

        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["use_modularity"] = 1234
            leiden(graph=graph, **args)

        args = good_args.copy()
        args["random_seed"] = 1234
        leiden(graph=graph, **args)
        args["random_seed"] = None
        leiden(graph=graph, **args)

        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["random_seed"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["random_seed"] = -1
            leiden(graph=graph, **args)

        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["is_weighted"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["weight_default"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["check_directed"] = "leiden"
            leiden(graph=graph, **args)

        # one extra parameter hierarchical needs
        with self.assertRaises(TypeError):
            args = good_args.copy()
            args["max_cluster_size"] = "leiden"
            hierarchical_leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["max_cluster_size"] = 0
            hierarchical_leiden(graph=graph, **args)

    def test_hierarchical(self):
        # most of leiden is tested in unit / integration tests in graspologic-native.
        # All we're trying to test through these unit tests are the python conversions
        # prior to calling, so type and value validation and that we got a result
        edges = _create_edge_list()
        results = hierarchical_leiden(edges, random_seed=1234)

        total_nodes = len([item for item in results if item.level == 0])

        partitions = HierarchicalCluster.final_hierarchical_clustering(results)
        self.assertEqual(total_nodes, len(partitions))
