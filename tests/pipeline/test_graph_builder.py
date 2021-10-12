# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import networkx as nx

from graspologic.pipeline import GraphBuilder


class TestGraphBuilder(unittest.TestCase):
    def test_simple_graph_builder(self):
        edges = [
            ("nick", "dax"),
            ("dax", "ben"),
            ("dax", "carolyn"),
            ("ben", "ali"),
        ]
        builder = GraphBuilder()  # undirected
        for source, target in edges:
            builder.add_edge(source, target)

        compressed_nx, old_to_new, new_to_old = builder.build()

        for node in compressed_nx.nodes():
            self.assertTrue(isinstance(node, int))

        expected_old_to_new = {"nick": 0, "dax": 1, "ben": 2, "carolyn": 3, "ali": 4}
        expected_new_to_old = ["nick", "dax", "ben", "carolyn", "ali"]
        expected_graph = nx.Graph()
        expected_graph.add_edge(0, 1, weight=1.0)
        expected_graph.add_edge(1, 2, weight=1.0)
        expected_graph.add_edge(1, 3, weight=1.0)
        expected_graph.add_edge(2, 4, weight=1.0)

        self.assertListEqual(
            expected_new_to_old, new_to_old, "The new to old index array should match"
        )
        self.assertDictEqual(
            expected_old_to_new, old_to_new, "The old to new dictionary should match"
        )
        nx.testing.assert_graphs_equal(compressed_nx, expected_graph)

    def test_weighted_graph_builder(self):
        edges = [("nick", "dax", 5.0), ("foo", "bar", 3.2)]
        expected_new_to_old = ["nick", "dax", "foo", "bar"]
        expected_old_to_new = {"nick": 0, "dax": 1, "foo": 2, "bar": 3}
        expected_nx = nx.Graph()
        expected_nx.add_edge(0, 1, weight=5.0)
        expected_nx.add_edge(3, 2, weight=3.2)
        builder = GraphBuilder()
        for source, target, weight in edges:
            builder.add_edge(source, target, weight)

        compressed, old_to_new, new_to_old = builder.build()
        nx.testing.assert_graphs_equal(expected_nx, compressed)
        self.assertListEqual(expected_new_to_old, new_to_old)
        self.assertDictEqual(expected_old_to_new, old_to_new)

    def test_summing_graph_builder(self):
        builder = GraphBuilder()
        builder.add_edge("dax", "nick", 5.0)
        builder.add_edge("nick", "dax", 3.2)
        builder.add_edge("dax", "nick")  # default weight of 1.0
        result_graph, _, _ = builder.build()
        self.assertEqual(9.2, result_graph[0][1]["weight"])

    def test_object_graph_builder(self):
        # why have you done this???
        first = ("dax", 1)
        second = ("nick", "cat")
        third = ("things", "in", "objects")
        fourth = ("arbitrary", 3)

        builder = GraphBuilder()
        builder.add_edge(first, second)
        builder.add_edge(third, fourth)
        builder.add_edge(first, fourth, 10.0)

        expected_new_to_old = [first, second, third, fourth]
        expected_old_to_new = {first: 0, second: 1, third: 2, fourth: 3}
        expected_nx = nx.Graph()
        expected_nx.add_edge(0, 1, weight=1.0)
        expected_nx.add_edge(2, 3, weight=1.0)
        expected_nx.add_edge(0, 3, weight=10.0)

        compressed, old_to_new, new_to_old = builder.build()

        self.assertListEqual(expected_new_to_old, new_to_old)
        self.assertDictEqual(expected_old_to_new, old_to_new)
        nx.testing.assert_graphs_equal(expected_nx, compressed)

    def test_int_graph_builder(self):
        edges = [(5000, 5001, 10.3), (13, 0, 11.1)]
        builder = GraphBuilder()
        for source, target, weight in edges:
            builder.add_edge(source, target, weight)
        expected_nx = nx.Graph()
        expected_nx.add_edge(0, 1, weight=10.3)
        expected_nx.add_edge(
            3, 2, weight=11.1
        )  # i put these out of order but it shouldn't matter
        expected_old_to_new = {5000: 0, 5001: 1, 13: 2, 0: 3}
        expected_new_to_old = [5000, 5001, 13, 0]

        compressed, old_to_new, new_to_old = builder.build()
        self.assertListEqual(new_to_old, expected_new_to_old)
        self.assertDictEqual(old_to_new, expected_old_to_new)
        nx.testing.assert_graphs_equal(expected_nx, compressed)

    def test_directed_graph_builder(self):
        edges = [("nick", "dax", 14.0), ("dax", "ben", 3.0), ("dax", "nick", 2.2)]
        builder = GraphBuilder(directed=True)
        for source, target, weight in edges:
            builder.add_edge(source, target, weight)

        expected_nx = nx.DiGraph()
        expected_nx.add_edge(0, 1, weight=14.0)
        expected_nx.add_edge(1, 2, weight=3.0)
        expected_nx.add_edge(1, 0, weight=2.2)
        expected_old_to_new = {"nick": 0, "dax": 1, "ben": 2}
        expected_new_to_old = ["nick", "dax", "ben"]

        compressed, old_to_new, new_to_old = builder.build()
        self.assertListEqual(new_to_old, expected_new_to_old)
        self.assertDictEqual(old_to_new, expected_old_to_new)
        nx.testing.assert_graphs_equal(expected_nx, compressed)

    def test_other_edge_attributes(self):
        builder = GraphBuilder()
        builder.add_edge(5000, 5001, 3.3, pandas="are so cute no really")
        builder.add_edge(5000, 5001, 10.3, sum_weight=False)
        builder.add_edge(13, 0, 11.1)
        builder.add_edge(0, 13, 13.3)

        expected_nx = nx.Graph()
        expected_nx.add_edge(0, 1, weight=10.3, pandas="are so cute no really")
        expected_nx.add_edge(2, 3, weight=24.4)

        expected_old_to_new = {5000: 0, 5001: 1, 13: 2, 0: 3}
        expected_new_to_old = [5000, 5001, 13, 0]

        compressed, old_to_new, new_to_old = builder.build()
        self.assertListEqual(new_to_old, expected_new_to_old)
        self.assertDictEqual(old_to_new, expected_old_to_new)
        nx.testing.assert_graphs_equal(expected_nx, compressed)
