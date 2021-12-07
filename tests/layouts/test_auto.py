# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import random
import unittest

import networkx as nx
import numpy

from graspologic.layouts.auto import _get_bounds, layout_umap


class TestAuto(unittest.TestCase):
    def test_get_bounds(self):
        y = numpy.array([(1, 2), (4, 5), (-1, -2), (10, -20)])
        minx, miny, maxx, maxy = _get_bounds(y)
        self.assertEqual(-1, minx)
        self.assertEqual(-20, miny)
        self.assertEqual(10, maxx)
        self.assertEqual(5, maxy)

    def test_layout_umap_string_node_ids(self):
        graph = nx.florentine_families_graph()

        for s, t in graph.edges():
            graph.add_edge(s, t, weight=1)

        _, node_positions = layout_umap(graph=graph)

        self.assertEqual(len(node_positions), len(graph.nodes()))

    def test_layout_umap_int_node_ids(self):
        graph = nx.florentine_families_graph()
        graph_int_node_ids = nx.Graph()
        ids_as_ints = dict()

        for s, t in graph.edges():
            if s not in ids_as_ints:
                ids_as_ints[s] = int(len(ids_as_ints.keys()))

            if t not in ids_as_ints:
                ids_as_ints[t] = int(len(ids_as_ints.keys()))

            graph_int_node_ids.add_edge(ids_as_ints[s], ids_as_ints[t], weight=1)

        _, node_positions = layout_umap(graph=graph_int_node_ids)

        self.assertEqual(len(node_positions), len(graph.nodes()))

    def test_layout_umap_directed_weighted(self):
        graph = nx.erdos_renyi_graph(10, 0.7, directed=True)

        for s, t in graph.edges():
            graph.edges[s, t]["weight"] = numpy.random.randint(1, 10)

        _, node_positions = layout_umap(graph=graph)

        self.assertEqual(len(node_positions), len(graph.nodes()))

    def test_layout_umap_directed_unweighted(self):
        graph = nx.erdos_renyi_graph(10, 0.7, directed=True)

        _, node_positions = layout_umap(graph=graph)

        self.assertEqual(len(node_positions), len(graph.nodes()))

    def test_exercise_approximate_prune(self):
        form = nx.erdos_renyi_graph(100, 0.7, directed=False)
        graph = nx.Graph()
        rng = random.Random(12345)
        for source, target in form.edges():
            graph.add_edge(str(source), str(target), weight=rng.uniform(0.0, 10.0))

        result_graph, positions = layout_umap(graph, max_edges=100)
        self.assertTrue(result_graph.number_of_edges() <= 100)


if __name__ == "__main__":
    unittest.main()
