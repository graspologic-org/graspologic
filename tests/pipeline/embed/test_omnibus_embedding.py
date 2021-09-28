# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import networkx as nx
import numpy as np
from beartype.roar import BeartypeCallHintPepParamException

from graspologic.pipeline.embed import omnibus_embedding_pairwise


class TestOmnibusEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = [nx.erdos_renyi_graph(1000, 0.2), nx.erdos_renyi_graph(1000, 0.5)]

    def _default_parameters(self):
        return {
            "dimensions": 100,
            "elbow_cut": None,
            "svd_solver_algorithm": "randomized",
            "svd_solver_iterations": 5,
            "svd_seed": None,
        }

    def test_argument_validation(self):
        with self.assertRaises(BeartypeCallHintPepParamException):
            omnibus_embedding_pairwise(graphs=[1])

        with self.assertRaises(ValueError):
            omnibus_embedding_pairwise(
                graphs=[nx.MultiDiGraph()], **self._default_parameters()
            )

        # dimensions
        dimensions = [None, 1.3, "1"]
        for dimension in dimensions:
            with self.assertRaises(BeartypeCallHintPepParamException):
                params = self._default_parameters()
                params["dimensions"] = dimension
                params["graphs"] = self.graphs
                omnibus_embedding_pairwise(**params)

        # elbow_cuts
        elbow_cuts = ["3", 1.3]
        for elbow_cut in elbow_cuts:
            with self.assertRaises(BeartypeCallHintPepParamException):
                params = self._default_parameters()
                params["elbow_cut"] = elbow_cut
                params["graphs"] = self.graphs
                omnibus_embedding_pairwise(**params)

        with self.assertRaises(BeartypeCallHintPepParamException):
            params = self._default_parameters()
            params["svd_solver_algorithm"] = 1
            params["graphs"] = self.graphs
            omnibus_embedding_pairwise(**params)

        with self.assertRaises(BeartypeCallHintPepParamException):
            params = self._default_parameters()
            params["svd_solver_algorithm"] = "sandwich"
            params["graphs"] = self.graphs
            omnibus_embedding_pairwise(**params)

        # svd_solver_iterations
        svd_solver_iterations = [None, "5", 5.1]
        for ssi in svd_solver_iterations:
            with self.assertRaises(BeartypeCallHintPepParamException):
                params = self._default_parameters()
                params["svd_solver_iterations"] = ssi
                params["graphs"] = self.graphs
                omnibus_embedding_pairwise(**params)

        # svd_seed
        svd_seeds = ["5", 5.1]
        for svd_seed in svd_seeds:
            with self.assertRaises(BeartypeCallHintPepParamException):
                params = self._default_parameters()
                params["svd_seed"] = svd_seed
                params["graphs"] = self.graphs
                omnibus_embedding_pairwise(**params)

    def test_omnibus_embedding_union_lcc_removes_isolates(self):
        g = nx.Graph()
        g.add_edge(1, 2, weight=1)
        g.add_edge(1, 3, weight=1)
        g.add_edge(2, 4, weight=1)
        g.add_node(5)

        g2 = g.copy()
        g2[2][4]["weight"] = 2
        g2.add_node(6)

        embeddings = omnibus_embedding_pairwise(graphs=[g, g2], dimensions=3)

        g_labels = embeddings[0][0].labels()
        g2_labels = embeddings[0][1].labels()

        np.testing.assert_array_equal(g_labels, g2_labels)
        np.testing.assert_array_equal(
            g_labels, [1, 2, 3, 4]
        )  # 5, 6 are not part of union LCC

    def test_omnibus_embedding_union_lcc_includes_nodes_from_union_lcc(self):
        g = nx.Graph()
        g.add_edge(1, 2, weight=1)
        g.add_edge(1, 3, weight=1)
        g.add_edge(2, 4, weight=1)
        g.add_edge(5, 6)  # not part of LCC

        g2 = g.copy()
        g2.add_edge(4, 5)  # create edge in union LCC, should include edge 5, 6 from g

        embeddings = omnibus_embedding_pairwise(graphs=[g, g2], dimensions=3)

        g_labels = embeddings[0][0].labels()
        g2_labels = embeddings[0][1].labels()

        np.testing.assert_array_equal(g_labels, g2_labels)
        # although 5 is not part of the LCC for g, it becomes part of the union LCC and will be included
        np.testing.assert_array_equal(g_labels, [1, 2, 3, 4, 5, 6])

    def test_omnibus_embedding_directed(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, weight=1)
        g.add_edge(1, 3, weight=1)
        g.add_edge(2, 4, weight=1)
        g.add_edge(5, 6)  # not part of LCC

        g2 = g.copy()
        g2.add_edge(4, 5)  # create edge in union LCC, should include edge 5, 6 from g

        embeddings = omnibus_embedding_pairwise(graphs=[g, g2], dimensions=3)

        g_labels = embeddings[0][0].labels()
        g2_labels = embeddings[0][1].labels()

        np.testing.assert_array_equal(g_labels, g2_labels)
        # although 5 is not part of the LCC for g, it becomes part of
        np.testing.assert_array_equal(g_labels, [1, 2, 3, 4, 5, 6])

    def test_omnibus_embedding_elbowcuts_none_returns_full_embedding(self):
        expected_dimensions = 100

        embeddings = omnibus_embedding_pairwise(
            graphs=self.graphs, dimensions=expected_dimensions, elbow_cut=None
        )

        for previous_embedding, current_embedding in embeddings:
            self.assertEqual(
                previous_embedding.embeddings().shape, (1000, expected_dimensions)
            )

    def test_omnibus_embedding_digraph_elbowcuts_none_returns_full_embedding(self):
        dimensions = 100
        expected_dimensions = dimensions * 2
        number_of_nodes = 1000

        g = nx.DiGraph()
        for i in range(number_of_nodes):
            g.add_edge(1, i, weight=1)

        g2 = g.copy()
        for i in range(number_of_nodes):
            g2.add_edge(i, 1, weight=i)

        embeddings = omnibus_embedding_pairwise(
            graphs=[g, g2], dimensions=dimensions, elbow_cut=None
        )

        for previous_embedding, current_embedding in embeddings:
            self.assertEqual(
                previous_embedding.embeddings().shape,
                (g.number_of_nodes(), expected_dimensions),
            )

    def test_omnibus_embedding_lse_digraph_elbowcuts_none_returns_full_embedding(self):
        dimensions = 100
        expected_dimensions = dimensions * 2
        number_of_nodes = 1000

        g = nx.DiGraph()
        for i in range(number_of_nodes):
            g.add_edge(1, i, weight=1)

        g2 = g.copy()
        for i in range(number_of_nodes):
            g2.add_edge(i, 1, weight=i)

        embeddings = omnibus_embedding_pairwise(
            graphs=[g, g2], dimensions=dimensions, elbow_cut=None, use_laplacian=True
        )

        for previous_embedding, current_embedding in embeddings:
            self.assertEqual(
                previous_embedding.embeddings().shape,
                (g.number_of_nodes(), expected_dimensions),
            )
