# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pytest
import scipy
from beartype.roar import BeartypeCallHintParamViolation

from graspologic.partition import (
    HierarchicalCluster,
    HierarchicalClusters,
    hierarchical_leiden,
    leiden,
)
from graspologic.partition.leiden import (
    _adjacency_matrix_to_edge_list,
    _edge_list_to_edge_list,
    _from_native,
    _IdentityMapper,
    _nx_to_edge_list,
)
from tests.utils import data_file


class TestHierarchicalCluster(unittest.TestCase):
    def test_from_native(self):
        with self.assertRaises(TypeError):
            _from_native(1, {"1": 1})

        # note: it is impossible to create a native instance of a HierarchicalCluster.  We will
        # test from_native indirectly through calling graspologic.partition.hierarchical_leiden()

    def test_final_hierarchical_clustering(self):
        hierarchical_clusters = HierarchicalClusters([
            HierarchicalCluster("1", 0, None, 0, False),
            HierarchicalCluster("2", 0, None, 0, False),
            HierarchicalCluster("3", 0, None, 0, False),
            HierarchicalCluster("4", 1, None, 0, True),
            HierarchicalCluster("5", 1, None, 0, True),
            HierarchicalCluster("1", 2, 0, 1, True),
            HierarchicalCluster("2", 2, 0, 1, True),
            HierarchicalCluster("3", 3, 0, 1, True),
        ])

        expected = {
            "1": 2,
            "2": 2,
            "3": 3,
            "4": 1,
            "5": 1,
        }
        self.assertEqual(
            expected,
            hierarchical_clusters.final_level_hierarchical_clustering(),
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
        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["starting_communities"] = 123
            leiden(graph=graph, **args)

        args = good_args.copy()
        args["starting_communities"] = None
        leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["extra_forced_iterations"] = 1234.003
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["extra_forced_iterations"] = -4003
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["resolution"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["resolution"] = 0
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["randomness"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["randomness"] = 0
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["use_modularity"] = 1234
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["trials"] = "hotdog"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["trials"] = 0
            leiden(graph=graph, **args)

        args = good_args.copy()
        args["random_seed"] = 1234
        leiden(graph=graph, **args)
        args["random_seed"] = None
        leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["random_seed"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["random_seed"] = -1
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["is_weighted"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["weight_default"] = "leiden"
            leiden(graph=graph, **args)

        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["check_directed"] = "leiden"
            leiden(graph=graph, **args)

        # one extra parameter hierarchical needs
        with self.assertRaises(BeartypeCallHintParamViolation):
            args = good_args.copy()
            args["max_cluster_size"] = "leiden"
            hierarchical_leiden(graph=graph, **args)

        with self.assertRaises(ValueError):
            args = good_args.copy()
            args["max_cluster_size"] = 0
            hierarchical_leiden(graph=graph, **args)

        cleared_partitions = good_args.copy()
        del cleared_partitions["starting_communities"]
        as_csr = nx.to_scipy_sparse_array(graph)
        partitions = leiden(graph=as_csr, **cleared_partitions)
        node_ids = partitions.keys()
        for node_id in node_ids:
            self.assertTrue(
                isinstance(
                    node_id, np.integer
                ),  # this is the preferred numpy typecheck
                f"{node_id} has {type(node_id)} should be an int",
            )

    def test_hierarchical(self):
        # most of leiden is tested in unit / integration tests in graspologic-native.
        # All we're trying to test through these unit tests are the python conversions
        # prior to calling, so type and value validation and that we got a result
        edges = _create_edge_list()
        results = hierarchical_leiden(edges, random_seed=1234)

        total_nodes = len([item for item in results if item.level == 0])

        partitions = results.final_level_hierarchical_clustering()
        self.assertEqual(total_nodes, len(partitions))

    # Github issue: 738
    def test_matching_return_types(self):
        graph = nx.erdos_renyi_graph(20, 0.4, seed=1234)
        partitions = leiden(graph)
        for node_id in partitions:
            self.assertTrue(isinstance(node_id, int))

    # Github issue: 901
    def test_hashable_nonstr_with_starting_communities(self):
        seed = 1234
        first_graph = nx.erdos_renyi_graph(20, 0.4, seed=seed)
        second_graph = nx.erdos_renyi_graph(21, 0.4, seed=seed)
        third_graph = nx.erdos_renyi_graph(19, 0.4, seed=seed)

        first_partitions = leiden(first_graph)
        second_partitions = leiden(second_graph, starting_communities=first_partitions)
        third_partitions = leiden(third_graph, starting_communities=second_partitions)


class TestLeidenIsolates(unittest.TestCase):
    """
    Tests to verify fix for Github issue: 803 - isolate nodes are dropped silently
    """

    def setUp(self) -> None:
        # prepare a graph with an isolate node
        self.graph: nx.Graph = nx.complete_graph(10)
        nodelist = sorted(self.graph.nodes)
        for node in nodelist[1:]:
            self.graph.remove_edge(0, node)

    def assert_isolate_not_in_result(self, partitions: Dict[str, int]):
        """verify that isolate node was not returned"""
        self.assertTrue(
            0 not in partitions, "the isolate node is not in the result from leiden"
        )
        self.assertTrue(
            3 in partitions, "a node that was not removed is in the result from leiden"
        )
        self.assertEqual(
            9,
            len(partitions),
            "the result contains all nodes in the connected component",
        )

    def assert_isolate_not_in_hierarchical_result(
        self, partitions: List[HierarchicalCluster]
    ):
        """verify that isolate node was not returned"""
        all_nodes = {p.node for p in partitions}

        self.assertTrue(
            0 not in all_nodes, "the isolate node is not in the result from leiden"
        )
        self.assertTrue(
            3 in all_nodes, "a node that was not removed is in the result from leiden"
        )
        self.assertEqual(
            9,
            len(all_nodes),
            "the result contains all nodes in the connected component",
        )

    def test_isolate_nodes_in_nx_graph_are_not_returned(self):
        self.assertEqual(
            10,
            len(self.graph.nodes),
            "the input graph contains all nodes including isolate",
        )

        with pytest.warns(UserWarning, match="isolate"):
            partitions = leiden(self.graph)

        self.assert_isolate_not_in_result(partitions)

        with pytest.warns(UserWarning, match="isolate"):
            hierarchical_partitions = hierarchical_leiden(self.graph)

        self.assert_isolate_not_in_hierarchical_result(hierarchical_partitions)

    def test_isolate_nodes_in_ndarray_are_not_returned(self):
        ndarray_adj_matrix = nx.to_numpy_array(self.graph)

        self.assertEqual(
            10,
            ndarray_adj_matrix.shape[0],
            "the input array contains all nodes including isolate",
        )

        with pytest.warns(UserWarning, match="isolate"):
            partitions = leiden(ndarray_adj_matrix)

        self.assert_isolate_not_in_result(partitions)

        with pytest.warns(UserWarning, match="isolate"):
            hierarchical_partitions = hierarchical_leiden(ndarray_adj_matrix)

        self.assert_isolate_not_in_hierarchical_result(hierarchical_partitions)

    def test_isolate_nodes_in_csr_array_are_not_returned(self):
        sparse_adj_matrix = nx.to_scipy_sparse_array(self.graph)

        self.assertEqual(
            10,
            sparse_adj_matrix.shape[0],
            "the input csr contains all nodes including isolate",
        )

        with pytest.warns(UserWarning, match="isolate"):
            partitions = leiden(sparse_adj_matrix)

        self.assert_isolate_not_in_result(partitions)

        with pytest.warns(UserWarning, match="isolate"):
            hierarchical_partitions = hierarchical_leiden(sparse_adj_matrix)

        self.assert_isolate_not_in_hierarchical_result(hierarchical_partitions)


def add_edges_to_graph(graph: nx.Graph) -> nx.Graph:
    graph.add_edge("nick", "dwayne", weight=1.0)
    graph.add_edge("nick", "dwayne", weight=3.0)
    graph.add_edge("dwayne", "nick", weight=2.2)
    graph.add_edge("dwayne", "ben", weight=4.2)
    graph.add_edge("ben", "dwayne", weight=0.001)
    return graph


class TestValidEdgeList(unittest.TestCase):
    def test_empty_edge_list(self):
        edges = []
        results = _edge_list_to_edge_list(
            edges=edges,
            identifier=_IdentityMapper(),
        )
        self.assertEqual([], results[1])

    def test_assert_list_does_not_contain_tuples(self):
        edges = ["invalid"]
        with self.assertRaises(BeartypeCallHintParamViolation):
            _edge_list_to_edge_list(
                edges=edges,
                identifier=_IdentityMapper(),
            )

    def test_assert_list_contains_misshapen_tuple(self):
        edges = [(1, 2, 1.0, 1.0)]
        with self.assertRaises(BeartypeCallHintParamViolation):
            _edge_list_to_edge_list(
                edges=edges,
                identifier=_IdentityMapper(),
            )

    def test_assert_wrong_types_in_tuples(self):
        edges = [(True, 4, "sandwich")]
        with self.assertRaises(BeartypeCallHintParamViolation):
            _edge_list_to_edge_list(
                edges=edges,
                identifier=_IdentityMapper(),
            )

        edges = [(True, False, 3.2)]
        _nodes, results = _edge_list_to_edge_list(
            edges=edges,
            identifier=_IdentityMapper(),
        )
        self.assertEqual([("True", "False", 3.2)], results)

    def test_empty_nx(self):
        expected = 0, []
        results = _nx_to_edge_list(
            graph=nx.Graph(),
            identifier=_IdentityMapper(),
            is_weighted=None,
            weight_attribute="weight",
            weight_default=1.0,
        )
        self.assertEqual(expected, results)
        with self.assertRaises(ValueError):
            _nx_to_edge_list(
                graph=nx.DiGraph(),
                identifier=_IdentityMapper(),
                is_weighted=None,
                weight_attribute="weight",
                weight_default=1.0,
            )
        with self.assertRaises(ValueError):
            _nx_to_edge_list(
                graph=nx.MultiGraph(),
                identifier=_IdentityMapper(),
                is_weighted=None,
                weight_attribute="weight",
                weight_default=1.0,
            )
        with self.assertRaises(ValueError):
            _nx_to_edge_list(
                graph=nx.MultiDiGraph(),
                identifier=_IdentityMapper(),
                is_weighted=None,
                weight_attribute="weight",
                weight_default=1.0,
            )

    def test_valid_nx(self):
        graph = add_edges_to_graph(nx.Graph())
        expected = [("nick", "dwayne", 2.2), ("dwayne", "ben", 0.001)]
        _, edges = _nx_to_edge_list(
            graph=graph,
            identifier=_IdentityMapper(),
            is_weighted=None,
            weight_attribute="weight",
            weight_default=1.0,
        )
        self.assertEqual(expected, edges)

    def test_unweighted_nx(self):
        graph = nx.Graph()
        graph.add_edge("dwayne", "nick")
        graph.add_edge("nick", "ben")

        with self.assertRaises(TypeError):
            _, edges = _nx_to_edge_list(
                graph=graph,
                identifier=_IdentityMapper(),
                is_weighted=True,
                weight_attribute="weight",
                weight_default=1.0,
            )

        _, edges = _nx_to_edge_list(
            graph=graph,
            identifier=_IdentityMapper(),
            is_weighted=False,
            weight_attribute="weight",
            weight_default=3.33333,
        )
        self.assertEqual(
            [("dwayne", "nick", 3.33333), ("nick", "ben", 3.33333)],
            edges,
        )

        graph.add_edge("salad", "sandwich", weight=100)
        _, edges = _nx_to_edge_list(
            graph=graph,
            identifier=_IdentityMapper(),
            is_weighted=False,
            weight_attribute="weight",
            weight_default=3.33333,
        )
        self.assertEqual(
            [
                ("dwayne", "nick", 3.33333),
                ("nick", "ben", 3.33333),
                ("salad", "sandwich", 100),
            ],
            edges,
        )

    def test_matrices(self):
        graph = add_edges_to_graph(nx.Graph())
        di_graph = add_edges_to_graph(nx.DiGraph())

        dense_undirected = nx.to_numpy_array(graph)
        dense_directed = nx.to_numpy_array(di_graph)

        sparse_undirected = nx.to_scipy_sparse_array(graph)
        sparse_directed = nx.to_scipy_sparse_array(di_graph)

        expected = [("0", "1", 2.2), ("1", "2", 0.001)]
        _, edges = _adjacency_matrix_to_edge_list(
            matrix=dense_undirected,
            identifier=_IdentityMapper(),
            check_directed=True,
            is_weighted=True,
            weight_default=1.0,
        )
        self.assertEqual(expected, edges)
        _, edges = _adjacency_matrix_to_edge_list(
            matrix=sparse_undirected,
            identifier=_IdentityMapper(),
            check_directed=True,
            is_weighted=True,
            weight_default=1.0,
        )
        self.assertEqual(expected, edges)

        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=dense_directed,
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=sparse_directed,
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

    def test_empty_adj_matrices(self):
        dense = np.array([])
        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=dense,
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

        sparse = scipy.sparse.csr_array([])
        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=sparse,
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

    def test_misshapen_matrices(self):
        data = [[3, 2, 0], [2, 0, 1]]  # this is utter gibberish
        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=np.array(data),
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )
        with self.assertRaises(ValueError):
            _adjacency_matrix_to_edge_list(
                matrix=scipy.sparse.csr_array(data),
                identifier=_IdentityMapper(),
                check_directed=True,
                is_weighted=True,
                weight_default=1.0,
            )

    def test_nx_graph_node_str_collision(self):
        graph = nx.Graph()
        graph.add_edge("1", 1, weight=1.0)
        with self.assertRaisesRegex(ValueError, "collision"):
            _nx_to_edge_list(
                graph=graph,
                identifier=_IdentityMapper(),
                is_weighted=True,
                weight_attribute="weight",
                weight_default=1.0,
            )

    def test_edgelist_node_str_collision(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            _edge_list_to_edge_list(edges=[("1", 1, 1.0)], identifier=_IdentityMapper())
