# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import graspologic as gc


class Node2VecEmbedTest(unittest.TestCase):
    def test_n2v_returns_same_labels_with_different_nodeid_types(self):
        random_seed = 1
        probability_matrix = np.array([[0.95, 0.01], [0.01, 0.95]])
        number_of_nodes_per_community = [20, 20]

        sbm_sample = gc.simulations.sbm(
            number_of_nodes_per_community, probability_matrix
        )
        sbm_graph = nx.from_numpy_array(sbm_sample)

        graph = nx.Graph()
        graph_as_strings = nx.Graph()
        for s, t in sbm_graph.edges():
            graph.add_edge(s, t, weight=1)
            graph_as_strings.add_edge(str(s), str(t), weight=1)

        original_embedding = gc.embed.node2vec_embed(graph, random_seed=random_seed)
        string_embedding = gc.embed.node2vec_embed(
            graph_as_strings, random_seed=random_seed
        )

        k = KMeans(n_clusters=2, random_state=random_seed)

        original_labels = k.fit_predict(original_embedding[0])
        string_labels = k.fit_predict(string_embedding[0])

        expected_labels = np.zeros(40, dtype=int)
        expected_labels[20:] = 1

        original_ari = adjusted_rand_score(original_labels, expected_labels)
        string_ari = adjusted_rand_score(string_labels, expected_labels)

        self.assertEqual(original_ari, string_ari)

    def test_n2v_directed_undirected_returns_same_clustering(self):
        probability_matrix = np.array([[0.95, 0.01], [0.01, 0.95]])
        number_of_nodes_per_community = [20, 20]

        sbm_sample = gc.simulations.sbm(
            number_of_nodes_per_community, probability_matrix
        )
        sbm_graph = nx.from_numpy_array(sbm_sample)

        graph = nx.Graph()
        graph_directed = nx.DiGraph()
        for s, t in sbm_graph.edges():
            graph.add_edge(s, t, weight=1)

            graph_directed.add_edge(s, t, weight=1)
            graph_directed.add_edge(t, s, weight=1)

        undirected_embedding = gc.embed.node2vec_embed(graph, random_seed=1)
        directed_embedding = gc.embed.node2vec_embed(graph_directed, random_seed=1)

        k = KMeans(n_clusters=2, random_state=1234)
        undirected_labels = k.fit_predict(undirected_embedding[0])
        k = KMeans(n_clusters=2, random_state=1234)
        directed_labels = k.fit_predict(directed_embedding[0])

        expected_labels = np.zeros(40, dtype=int)
        expected_labels[20:] = 1

        undirected_ari = adjusted_rand_score(undirected_labels, expected_labels)
        directed_ari = adjusted_rand_score(directed_labels, expected_labels)

        self.assertEqual(undirected_ari, directed_ari)

    def test_node2vec_embed(self):
        g = nx.florentine_families_graph()

        for s, t in g.edges():
            g.add_edge(s, t, weight=1)

        embedding = gc.embed.node2vec_embed(g, random_seed=1)

        embedding2 = gc.embed.node2vec_embed(g, random_seed=1)

        np.testing.assert_array_equal(embedding[0], embedding2[0])

    def test_node2vec_embedding_florentine_graph_correct_shape_is_returned(self):
        graph = nx.florentine_families_graph()
        for s, t in graph.edges():
            graph.add_edge(s, t, weight=1)

        model = gc.embed.node2vec_embed(graph)
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

    def test_node2vec_embedding_unweighted_florentine_graph_correct_shape_is_returned(
        self,
    ):
        graph = nx.florentine_families_graph()

        model = gc.embed.node2vec_embed(graph)
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

    def test_node2vec_same_labels_are_returned(self):
        graph = nx.florentine_families_graph()
        node_ids = list(graph.nodes())

        embedding, labels = gc.embed.node2vec_embed(graph)

        for i in range(len(node_ids)):
            self.assertEqual(node_ids[i], labels[i])

    def test_node2vec_embedding_barbell_graph_correct_shape_is_returned(self):
        graph = nx.barbell_graph(25, 2)
        for s, t in graph.edges():
            graph.add_edge(s, t, weight=1)

        model = gc.embed.node2vec_embed(graph)
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

        g = gc.embed.n2v._Node2VecGraph(nx.Graph(), 1, 1)
        w = g._get_walk_length_interpolated(
            degree=0, percentiles=[1, 2, 3, 4, 10, 100], max_walk_length=10
        )

        self.assertEqual(w, expected_walk_length)

    def test_get_walk_length_higher_default_to_walk_length(self):
        expected_walk_length = 100

        g = gc.embed.n2v._Node2VecGraph(nx.Graph(), 1, 1)
        w = g._get_walk_length_interpolated(
            degree=10,
            percentiles=[2, 3, 4, 5, 6, 7, 8, 9],
            max_walk_length=expected_walk_length,
        )

        self.assertEqual(w, expected_walk_length)

    def test_get_walk_length_in_middle_selects_interpolated_bucket(self):
        expected_walk_length = 5

        g = gc.embed.n2v._Node2VecGraph(nx.Graph(), 1, 1)
        w = g._get_walk_length_interpolated(
            degree=5, percentiles=[2, 3, 4, 5, 6, 7, 8, 9], max_walk_length=10
        )

        self.assertEqual(w, expected_walk_length)

    def test_random_state_is_initialized_in_constructor(self):
        graph = nx.barbell_graph(25, 2)
        start_node = "0"

        n2v = gc.embed.n2v._Node2VecGraph(graph, 1, 1)
        n2v._preprocess_transition_probabilities()
        walk = n2v.node2vec_walk(5, start_node, None)
        self.assertGreater(len(walk), 0)
