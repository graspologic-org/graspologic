# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import networkx as nx
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintPepParamException

import graspologic.utils
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.pipeline.embed import adjacency_spectral_embedding
from tests.utils import data_file


class TestAdjacencySpectralEmbedding(unittest.TestCase):
    # despite the name, we're not actually testing ase - that's already covered in
    # tests/test_spectral_embed
    # instead, we're going to test type and range checks, the elbow finding,
    # and at the least that a fixed dimension embed request from pipeline
    # matches a fixed dimension embed request of
    # graspologic.embed.AdjacencySpectralEmbed

    def setUp(self) -> None:
        self.graph = nx.erdos_renyi_graph(1000, 0.2)
        self.default_parameters = {
            "dimensions": 100,
            "elbow_cut": None,
            "svd_solver_algorithm": "randomized",
            "svd_solver_iterations": 5,
            "svd_seed": None,
        }

    @staticmethod
    def parameters():
        return {
            "dimensions": 100,
            "elbow_cut": None,
            "svd_solver_algorithm": "randomized",
            "svd_solver_iterations": 5,
            "svd_seed": None,
        }

    def test_argument_validation(self):
        # graph types
        with self.assertRaises(BeartypeCallHintPepParamException):
            adjacency_spectral_embedding(
                graph=np.array([[1, 2], [2, 1]]), **self.default_parameters
            )
        with self.assertRaises(ValueError):
            adjacency_spectral_embedding(
                graph=nx.MultiDiGraph(), **self.default_parameters
            )

        # dimensions
        dimensions = [None, 1.3, "1"]
        for dimension in dimensions:
            with self.assertRaises(BeartypeCallHintPepParamException):
                params = TestAdjacencySpectralEmbedding.parameters()
                params["dimensions"] = dimension
                params["graph"] = self.graph
                adjacency_spectral_embedding(**params)

        # elbow_cuts
        elbow_cuts = ["3", 1.3]
        for elbow_cut in elbow_cuts:
            with self.assertRaises(BeartypeCallHintPepParamException):
                params = TestAdjacencySpectralEmbedding.parameters()
                params["elbow_cut"] = elbow_cut
                params["graph"] = self.graph
                adjacency_spectral_embedding(**params)

        with self.assertRaises(BeartypeCallHintPepParamException):
            params = TestAdjacencySpectralEmbedding.parameters()
            params["svd_solver_algorithm"] = 1
            params["graph"] = self.graph
            adjacency_spectral_embedding(**params)

        with self.assertRaises(BeartypeCallHintPepParamException):
            params = TestAdjacencySpectralEmbedding.parameters()
            params["svd_solver_algorithm"] = "sandwich"
            params["graph"] = self.graph
            adjacency_spectral_embedding(**params)

        # svd_solver_iterations
        svd_solver_iterations = [None, "5", 5.1]
        for ssi in svd_solver_iterations:
            with self.assertRaises(BeartypeCallHintPepParamException):
                params = TestAdjacencySpectralEmbedding.parameters()
                params["svd_solver_iterations"] = ssi
                params["graph"] = self.graph
                adjacency_spectral_embedding(**params)

        # svd_seed
        svd_seeds = ["5", 5.1]
        for svd_seed in svd_seeds:
            with self.assertRaises(BeartypeCallHintPepParamException):
                params = TestAdjacencySpectralEmbedding.parameters()
                params["svd_seed"] = svd_seed
                params["graph"] = self.graph
                adjacency_spectral_embedding(**params)

    def test_unweighted_graph_warning(self):
        graph = self.graph
        with pytest.warns(UserWarning):
            adjacency_spectral_embedding(graph)

    def test_dimensions(self):
        graph = self.graph.copy()
        sparse = nx.to_scipy_sparse_matrix(graph)
        ranked = graspologic.utils.pass_to_ranks(sparse)
        ase = AdjacencySpectralEmbed(n_components=100, n_elbows=None, svd_seed=1234)
        core_response = ase.fit_transform(ranked)

        embedding = adjacency_spectral_embedding(self.graph.copy(), svd_seed=1234)
        np.testing.assert_array_almost_equal(core_response, embedding.embeddings())

    def test_elbow_cuts(self):
        # smoke test, unsure how to validate the elbow is cut at the right spot
        # for this data
        graph = nx.Graph()
        digraph = nx.DiGraph()

        with open(data_file("large-graph.csv"), "r") as graph_io:
            for line in graph_io:
                source, target, weight = line.strip().split(",")
                prev_weight = graph.get_edge_data(source, target, default={}).get(
                    "weight", 0.0
                )
                graph.add_edge(source, target, weight=float(weight) + prev_weight)
                digraph.add_edge(source, target, weight=float(weight))

        dimensions = 100
        results = adjacency_spectral_embedding(
            graph, dimensions=dimensions, elbow_cut=2, svd_seed=1234
        )
        self.assertTrue(results.embeddings().shape[1] < dimensions)
        results = adjacency_spectral_embedding(
            digraph, dimensions=dimensions, elbow_cut=2, svd_seed=1234
        )
        self.assertTrue(results.embeddings().shape[1] < dimensions * 2)
        results = adjacency_spectral_embedding(
            digraph, dimensions=dimensions, elbow_cut=None, svd_seed=1234
        )
        self.assertTrue(results.embeddings().shape[1] == dimensions * 2)
