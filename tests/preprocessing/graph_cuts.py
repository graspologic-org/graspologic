# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import networkx as nx
import numpy as np
from testfixtures import LogCapture

from graspologic.preprocessing import graph_cuts

from ..utils import data_file


def _get_florentine_graph():
    graph = nx.florentine_families_graph()

    for s, t in graph.edges():
        graph.add_edge(s, t, weight=1)

    return graph


def _get_actor_graph():
    graph = nx.Graph()
    with open(data_file("actor_bipartite_graph.csv")) as source_iterator:
        next(source_iterator)  # skip the header
        for line in source_iterator:
            source, target, _ = line.strip().split(",")
            graph.add_edge(source, target, weight=1)
    return graph


def _get_toy_graph():
    graph = nx.Graph()
    graph.add_edge(1, 2, weight=3)
    graph.add_edge(4, 5, weight=3)
    graph.add_edge("a", "b", weight=5)
    graph.add_edge("nick", "dwayne", weight=7)
    return graph


class TestBetweennessCentralityHistogram(unittest.TestCase):
    def test_histogram_by_bin_count(self):
        graph = _get_florentine_graph()
        defined_histogram = graph_cuts.histogram_betweenness_centrality(graph, 10)
        self.assertEqual(10, len(defined_histogram.histogram))
        self.assertEqual(11, len(defined_histogram.bin_edges))

    def test_histogram_by_edge_bins(self):
        graph = _get_florentine_graph()
        defined_histogram = graph_cuts.histogram_betweenness_centrality(
            graph, [0.0, 0.03, 2.0]
        )
        self.assertEqual(2, len(defined_histogram.histogram))
        self.assertEqual(3, len(defined_histogram.bin_edges))
        np.testing.assert_array_equal(
            np.array([0.0, 0.03, 2.0], dtype=np.dtype(float)),
            defined_histogram.bin_edges,
        )

    def test_histogram_by_auto(self):
        graph = _get_florentine_graph()
        defined_histogram = graph_cuts.histogram_betweenness_centrality(graph, "auto")
        self.assertEqual(6, len(defined_histogram.histogram))
        self.assertEqual(7, len(defined_histogram.bin_edges))


class TestBetweennessCentralityCut(unittest.TestCase):
    def test_cut_none(self):
        graph = _get_florentine_graph()
        expected_graph_nodes = 14
        expected_graph_edges = 14

        result = graph_cuts.cut_vertices_by_betweenness_centrality(
            graph, 0.4, graph_cuts.LARGER_THAN_EXCLUSIVE
        )
        self.assertEqual(expected_graph_edges, len(result.edges))
        self.assertEqual(expected_graph_nodes, len(result.nodes))

    def test_cut_all(self):
        graph = _get_florentine_graph()

        result = graph_cuts.cut_vertices_by_betweenness_centrality(
            graph, 0.5, graph_cuts.SMALLER_THAN_INCLUSIVE
        )
        self.assertEqual(0, len(result.edges))
        self.assertEqual(1, len(result.nodes))

    def test_cut_less_than_inclusive(self):
        graph = _get_florentine_graph()

        result = graph_cuts.cut_vertices_by_betweenness_centrality(
            graph, 0.0166525071464909, graph_cuts.SMALLER_THAN_INCLUSIVE
        )
        self.assertEqual(11, len(result.nodes))

    def test_cut_less_than_exclusive(self):
        graph = _get_florentine_graph()

        result = graph_cuts.cut_vertices_by_betweenness_centrality(
            graph, 0.017, graph_cuts.SMALLER_THAN_EXCLUSIVE
        )
        self.assertEqual(11, len(result.nodes))

    def test_cut_greater_than_inclusive(self):
        graph = _get_florentine_graph()

        result = graph_cuts.cut_vertices_by_betweenness_centrality(
            graph, 0.01665250714649088, graph_cuts.LARGER_THAN_INCLUSIVE
        )
        self.assertEqual(4, len(result.nodes))

    def test_cut_greater_than_exclusive(self):
        graph = _get_florentine_graph()

        result = graph_cuts.cut_vertices_by_betweenness_centrality(
            graph, 0.017, graph_cuts.LARGER_THAN_EXCLUSIVE
        )
        self.assertEqual(4, len(result.nodes))


class TestDegreeCentralityHistogram(unittest.TestCase):
    def test_histogram_by_bin_count(self):
        graph = _get_actor_graph()
        defined_histogram = graph_cuts.histogram_degree_centrality(graph, 10)
        self.assertEqual(10, len(defined_histogram.histogram))
        self.assertEqual(11, len(defined_histogram.bin_edges))

    def test_histogram_by_edge_bins(self):
        graph = _get_actor_graph()
        defined_histogram = graph_cuts.histogram_degree_centrality(
            graph, [0.0, 0.03, 2.0]
        )
        self.assertEqual(2, len(defined_histogram.histogram))
        self.assertEqual(3, len(defined_histogram.bin_edges))
        np.testing.assert_array_equal(
            np.array([0.0, 0.03, 2.0], dtype=np.dtype(float)),
            defined_histogram.bin_edges,
        )

    def test_histogram_by_auto(self):
        graph = _get_actor_graph()
        defined_histogram = graph_cuts.histogram_degree_centrality(graph, "auto")
        self.assertEqual(5, len(defined_histogram.histogram))
        self.assertEqual(6, len(defined_histogram.bin_edges))


class TestDegreeCentralityCut(unittest.TestCase):
    def test_cut_none(self):
        graph = _get_actor_graph()
        expected_graph_nodes = len(graph.nodes)
        expected_graph_edges = len(graph.edges)

        result = graph_cuts.cut_vertices_by_degree_centrality(
            graph, 0.5, graph_cuts.LARGER_THAN_EXCLUSIVE
        )
        self.assertEqual(expected_graph_edges, len(result.edges))
        self.assertEqual(expected_graph_nodes, len(result.nodes))

    def test_cut_all(self):
        graph = _get_actor_graph()

        result = graph_cuts.cut_vertices_by_degree_centrality(
            graph, 0.5, graph_cuts.SMALLER_THAN_INCLUSIVE
        )
        self.assertEqual(0, len(result.edges))
        self.assertEqual(0, len(result.nodes))

    def test_cut_less_than_inclusive(self):
        graph = _get_actor_graph()

        result = graph_cuts.cut_vertices_by_degree_centrality(
            graph, 0.01665, graph_cuts.SMALLER_THAN_INCLUSIVE
        )
        self.assertEqual(11, len(result.nodes))

    def test_cut_less_than_exclusive(self):
        graph = _get_actor_graph()

        result = graph_cuts.cut_vertices_by_degree_centrality(
            graph, 0.01666666666666668, graph_cuts.SMALLER_THAN_EXCLUSIVE
        )
        self.assertEqual(11, len(result.nodes))

    def test_cut_greater_than_inclusive(self):
        graph = _get_actor_graph()

        result = graph_cuts.cut_vertices_by_degree_centrality(
            graph, 0.139280, graph_cuts.LARGER_THAN_INCLUSIVE
        )
        self.assertEqual(6, len(result.nodes))

    def test_cut_greater_than_exclusive(self):
        graph = _get_actor_graph()

        result = graph_cuts.cut_vertices_by_degree_centrality(
            graph, 0.1392857142857144, graph_cuts.LARGER_THAN_EXCLUSIVE
        )
        self.assertEqual(6, len(result.nodes))


class TestEdgeWeights(unittest.TestCase):
    def test_histogram_from_graph(self):
        with LogCapture() as log_capture:
            graph = nx.Graph()
            graph.add_edge(1, 2, weight=3)
            graph.add_edge(4, 5, weight=3)
            graph.add_edge("a", "b", weight=5)
            graph.add_edge("nick", "dwayne")
            expected = graph_cuts.DefinedHistogram(
                histogram=np.array([2, 1]), bin_edges=np.array([3, 4, 5])
            )
            result = graph_cuts.histogram_edge_weight(graph, 2)
            np.testing.assert_array_equal(expected.histogram, result.histogram)
            np.testing.assert_array_equal(expected.bin_edges, result.bin_edges)

            # check logger is logging things correctly since it is an important part of this function
            # by proxy this also checks that edges_by_weight is called
            log_capture.check(
                (
                    "graspologic.preprocessing.graph_cuts",
                    "WARNING",
                    "Graph contains 1 edges with no weight. Histogram excludes these values.",
                )
            )

    def test_make_cuts_larger_than_inclusive(self):
        graph = _get_toy_graph()

        updated_graph = graph_cuts.cut_edges_by_weight(
            graph, 5, graph_cuts.LARGER_THAN_INCLUSIVE, prune_isolates=True
        )
        self.assertEqual(2, len(updated_graph.edges))
        self.assertEqual(4, len(updated_graph.nodes))
        self.assertEqual(3, updated_graph[1][2]["weight"])
        self.assertEqual(3, updated_graph[5][4]["weight"])

    def test_make_cuts_larger_than_exclusive(self):
        graph = _get_toy_graph()

        updated_graph = graph_cuts.cut_edges_by_weight(
            graph, 5, graph_cuts.LARGER_THAN_EXCLUSIVE, prune_isolates=True
        )
        self.assertEqual(3, len(updated_graph.edges))
        self.assertEqual(6, len(updated_graph.nodes))
        self.assertEqual(3, updated_graph[1][2]["weight"])
        self.assertEqual(3, updated_graph[5][4]["weight"])
        self.assertEqual(5, updated_graph["b"]["a"]["weight"])

    def test_make_cuts_smaller_than_inclusive(self):
        graph = _get_toy_graph()

        updated_graph = graph_cuts.cut_edges_by_weight(
            graph, 5, graph_cuts.SMALLER_THAN_INCLUSIVE, prune_isolates=True
        )
        self.assertEqual(1, len(updated_graph.edges))
        self.assertEqual(2, len(updated_graph.nodes))
        self.assertEqual(7, updated_graph["nick"]["dwayne"]["weight"])

    def test_make_cuts_smaller_than_exclusive(self):
        graph = _get_toy_graph()

        updated_graph = graph_cuts.cut_edges_by_weight(
            graph, 5, graph_cuts.SMALLER_THAN_EXCLUSIVE, prune_isolates=True
        )
        self.assertEqual(2, len(updated_graph.edges))
        self.assertEqual(4, len(updated_graph.nodes))
        self.assertEqual(5, updated_graph["a"]["b"]["weight"])
        self.assertEqual(7, updated_graph["nick"]["dwayne"]["weight"])

    def test_make_cuts_smaller_than_exclusive_no_prune_isolates(self):
        graph = _get_toy_graph()

        updated_graph = graph_cuts.cut_edges_by_weight(
            graph, 5, graph_cuts.SMALLER_THAN_EXCLUSIVE
        )
        self.assertEqual(2, len(updated_graph.edges))
        self.assertEqual(8, len(updated_graph.nodes))
        self.assertEqual(5, updated_graph["a"]["b"]["weight"])
        self.assertEqual(7, updated_graph["nick"]["dwayne"]["weight"])
        self.assertIn(1, updated_graph)
        self.assertIn(2, updated_graph)
        self.assertIn(4, updated_graph)
        self.assertIn(5, updated_graph)

    def test_cut_all(self):
        graph = _get_toy_graph()

        updated_graph = graph_cuts.cut_edges_by_weight(
            graph, 7, graph_cuts.SMALLER_THAN_INCLUSIVE, prune_isolates=True
        )
        self.assertEqual(0, len(updated_graph.edges))
        self.assertEqual(0, len(updated_graph.nodes))

    def test_cut_none(self):
        graph = _get_toy_graph()

        updated_graph = graph_cuts.cut_edges_by_weight(
            graph, 7, graph_cuts.LARGER_THAN_EXCLUSIVE, prune_isolates=True
        )
        self.assertEqual(4, len(updated_graph.edges))
        self.assertEqual(8, len(updated_graph.nodes))

    def test_broken_make_cuts(self):
        graph = _get_toy_graph()
        with self.assertRaises(ValueError):
            graph_cuts.cut_edges_by_weight(graph, 5, None)
