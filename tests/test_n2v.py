# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
import networkx as nx
import numpy as np
import graspologic.embed.n2v as n2v
from graspologic.embed.n2v import _Node2VecGraph


class Node2VecEmbedTest(unittest.TestCase):
    def test_node2vec_embedding_correct_shape_is_returned(self):
        import io

        graph = nx.read_edgelist(
            io.StringIO(_edge_list), nodetype=int, create_using=nx.DiGraph()
        )

        model = n2v.node2vec_embed(graph)
        model_matrix: np.ndarray = model[0]
        vocab_list = model[1]
        self.assertIsNotNone(model)
        self.assertIsNotNone(model[0])
        self.assertIsNotNone(model[1])

        # model matrix should be 34 x 128
        self.assertEqual(model_matrix.shape[0], 34)
        self.assertEqual(model_matrix.shape[1], 128)

        # vocab list should have exactly 34 elements
        self.assertEqual(len(vocab_list), 34)

    def test_node2vec_embedding_florentine_graph_correct_shape_is_returned(self):
        graph = nx.florentine_families_graph()
        for s, t in graph.edges():
            graph.add_edge(s, t, weight=1)

        model = n2v.node2vec_embed(graph)
        model_matrix: np.ndarray = model[0]
        vocab_list = model[1]
        self.assertIsNotNone(model)
        self.assertIsNotNone(model[0])
        self.assertIsNotNone(model[1])

        # model matrix should be 34 x 128
        self.assertEqual(model_matrix.shape[0], 15)
        self.assertEqual(model_matrix.shape[1], 128)

        # vocab list should have exactly 34 elements
        self.assertEqual(len(vocab_list), 15)

    def test_node2vec_embedding_barbell_graph_correct_shape_is_returned(self):
        graph = nx.barbell_graph(25, 2)
        for s, t in graph.edges():
            graph.add_edge(s, t, weight=1)

        model = n2v.node2vec_embed(graph)
        model_matrix: np.ndarray = model[0]
        vocab_list = model[1]
        self.assertIsNotNone(model)
        self.assertIsNotNone(model[0])
        self.assertIsNotNone(model[1])

        # model matrix should be 34 x 128
        self.assertEqual(model_matrix.shape[0], 52)
        self.assertEqual(model_matrix.shape[1], 128)

        # vocab list should have exactly 34 elements
        self.assertEqual(len(vocab_list), 52)

    def test_get_walk_length_lower_defaults_to_1(self):
        expected_walk_length = 1

        g = _Node2VecGraph(nx.Graph(), 1, 1)
        w = g._get_walk_length_interpolated(
            degree=0, percentiles=[1, 2, 3, 4, 10, 100], max_walk_length=10
        )

        self.assertEqual(w, expected_walk_length)

    def test_get_walk_length_higher_default_to_walk_length(self):
        expected_walk_length = 100

        g = _Node2VecGraph(nx.Graph(), 1, 1)
        w = g._get_walk_length_interpolated(
            degree=10,
            percentiles=[2, 3, 4, 5, 6, 7, 8, 9],
            max_walk_length=expected_walk_length,
        )

        self.assertEqual(w, expected_walk_length)

    def test_get_walk_length_in_middle_selects_interpolated_bucket(self):
        expected_walk_length = 5

        g = _Node2VecGraph(nx.Graph(), 1, 1)
        w = g._get_walk_length_interpolated(
            degree=5, percentiles=[2, 3, 4, 5, 6, 7, 8, 9], max_walk_length=10
        )

        self.assertEqual(w, expected_walk_length)


_edge_list = """
1 32
1 22
1 20
1 18
1 14
1 13
1 12
1 11
1 9
1 8
1 7
1 6
1 5
1 4
1 3
1 2
2 31
2 22
2 20
2 18
2 14
2 8
2 4
2 3
3 14
3 9
3 10
3 33
3 29
3 28
3 8
3 4
4 14
4 13
4 8
5 11
5 7
6 17
6 11
6 7
7 17
9 34
9 33
9 33
10 34
14 34
15 34
15 33
16 34
16 33
19 34
19 33
20 34
21 34
21 33
23 34
23 33
24 30
24 34
24 33
24 28
24 26
25 32
25 28
25 26
26 32
27 34
27 30
28 34
29 34
29 32
30 34
30 33
31 34
31 33
32 34
32 33
33 34
"""
