# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import os
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import graspologic as gs


class TestImportGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # simple ERxN graph
        n = 15
        p = 0.5
        cls.A = np.zeros((n, n))
        nedge = int(round(n * n * p))
        np.put(
            cls.A,
            np.random.choice(np.arange(0, n * n), size=nedge, replace=False),
            np.random.normal(size=nedge),
        )

    def test_graphin(self):
        G = nx.from_numpy_array(self.A)
        np.testing.assert_array_equal(nx.to_numpy_array(G), gs.utils.import_graph(G))

    def test_npin(self):
        np.testing.assert_array_equal(self.A, gs.utils.import_graph(self.A))

    def test_wrongtypein(self):
        a = 5
        with self.assertRaises(TypeError):
            gs.utils.import_graph(a)
        with self.assertRaises(TypeError):
            gs.utils.import_graph(None)

    def test_nonsquare(self):
        non_square = np.hstack((self.A, self.A))
        with self.assertRaises(ValueError):
            gs.utils.import_graph(non_square)


class TestImportEdgelist(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        n = 10
        p = 0.5
        wt = np.random.exponential
        wtargs = dict(scale=4)

        np.random.seed(1)

        cls.A = gs.simulations.er_np(n, p)
        cls.B = gs.simulations.er_np(n, p, wt=wt, wtargs=wtargs)

        G_A = nx.from_numpy_array(cls.A)
        G_B = nx.from_numpy_array(cls.B)
        G_B = nx.relabel_nodes(G_B, lambda x: x + 10)  # relabel nodes to go from 10-19.

        cls.root = str(cls.tmpdir.name)
        cls.A_path = os.path.join(cls.root, "A_unweighted.edgelist")
        cls.B_path = os.path.join(cls.root, "B.edgelist")

        nx.write_edgelist(G_A, cls.A_path, data=False)
        nx.write_weighted_edgelist(G_B, cls.B_path)

    def test_in(self):
        A_from_edgelist = gs.utils.import_edgelist(self.A_path)
        B_from_edgelist = gs.utils.import_edgelist(self.B_path)

        np.testing.assert_allclose(A_from_edgelist, self.A)
        np.testing.assert_allclose(B_from_edgelist, self.B)

    def test_in_Path_obj(self):
        A_from_edgelist = gs.utils.import_edgelist(Path(self.A_path))
        B_from_edgelist = gs.utils.import_edgelist(Path(self.B_path))

        np.testing.assert_allclose(A_from_edgelist, self.A)
        np.testing.assert_allclose(B_from_edgelist, self.B)

    def test_multiple_in(self):
        graphs = gs.utils.import_edgelist(self.root)
        A = np.zeros((20, 20))
        A[:10, :10] = self.A

        B = np.zeros((20, 20))
        B[10:, 10:] = self.B

        self.assertEqual(len(graphs), 2)
        self.assertTrue(all(graph.shape == (20, 20) for graph in graphs))
        np.testing.assert_allclose(graphs[0], A)
        np.testing.assert_allclose(graphs[1], B)

    def test_wrongtypein(self):
        path = 5
        with self.assertRaises(TypeError):
            gs.utils.import_edgelist(path)
        with self.assertRaises(TypeError):
            gs.utils.import_edgelist(None)

    def test_vertices(self):
        expected_vertices_A = np.arange(0, 10)
        expected_vertices_B = np.arange(10, 20)

        _, A_vertices = gs.utils.import_edgelist(self.A_path, return_vertices=True)
        _, B_vertices = gs.utils.import_edgelist(self.B_path, return_vertices=True)

        np.testing.assert_allclose(expected_vertices_A, A_vertices)
        np.testing.assert_allclose(expected_vertices_B, B_vertices)

    def test_no_graphs_found(self):
        path = str(self.root + "invalid_edgelist.edgelist")
        with self.assertRaises(ValueError):
            gs.utils.import_edgelist(path)

    def test_bad_delimiter(self):
        delimiter = ","
        with pytest.warns(UserWarning):
            graphs = gs.utils.import_edgelist(self.root, delimiter=delimiter)
