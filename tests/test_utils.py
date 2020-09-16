# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
import numpy as np
import networkx as nx
from graspy.utils import utils as gus
from math import sqrt
from numpy.testing import assert_equal


class TestInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        np.testing.assert_array_equal(nx.to_numpy_array(G), gus.import_graph(G))

    def test_npin(self):
        np.testing.assert_array_equal(self.A, gus.import_graph(self.A))

    def test_wrongtypein(self):
        a = 5
        with self.assertRaises(TypeError):
            gus.import_graph(a)
        with self.assertRaises(TypeError):
            gus.import_graph(None)


class TestToLaplace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        cls.B = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 0]])

    def test_to_laplace_IDAD(self):
        expected_L_normed = [
            [1, -1 / (sqrt(2)), 0],
            [-1 / (sqrt(2)), 1, -1 / (sqrt(2))],
            [0, -1 / (sqrt(2)), 1],
        ]

        L_normed = gus.to_laplace(self.A, form="I-DAD")
        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

    def test_to_laplace_DAD(self):
        expected_L_normed = [
            [0, 1 / sqrt(2), 0],
            [1 / sqrt(2), 0, 1 / sqrt(2)],
            [0, 1 / sqrt(2), 0],
        ]

        L_normed = gus.to_laplace(self.A, form="DAD")

        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

    def test_to_laplace_RDAD(self):
        expected_L_normed = [
            [0, 3 / sqrt(70), 0],
            [3 / sqrt(70), 0, 3 / sqrt(70)],
            [0, 3 / sqrt(70), 0],
        ]

        L_normed = gus.to_laplace(self.A, form="R-DAD")

        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

    def test_to_laplace_regularizer_kwarg(self):
        expected_L_normed = [
            [0, 1 / sqrt(6), 0],
            [1 / sqrt(6), 0, 1 / sqrt(6)],
            [0, 1 / sqrt(6), 0],
        ]
        L_normed = gus.to_laplace(self.A, form="R-DAD", regularizer=1.0)

        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

    def test_to_laplace_symmetric(self):
        L_normed = gus.to_laplace(self.A, form="DAD")

        self.assertTrue(gus.is_symmetric(L_normed))

    def test_to_laplace_unsuported(self):
        with self.assertRaises(TypeError):
            gus.to_laplace(self.A, form="MOM")

    def test_to_laplace_unsuported_regularizer(self):
        with self.assertRaises(TypeError):
            gus.to_laplace(self.A, form="R-DAD", regularizer="2")
        with self.assertRaises(TypeError):
            gus.to_laplace(self.A, form="R-DAD", regularizer=[1, 2, 3])
        with self.assertRaises(ValueError):
            gus.to_laplace(self.A, form="R-DAD", regularizer=-1.0)

    def test_to_laplace_directed(self):
        expected_L_normed = [
            [0, 1 / 5, sqrt(5) / 10, 0.2],
            [0, 0, 0, sqrt(15) / 15],
            [0, sqrt(5) / 10, 0, sqrt(5) / 10],
            [0, sqrt(5) / 10, 0.25, 0],
        ]
        L_normed = gus.to_laplace(self.B, form="R-DAD")
        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))


class TestChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def test_is_unweighted(self):
        B = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1.0, 0, 0], [1, 0, 1, 0]])
        self.assertTrue(gus.is_unweighted(B))
        self.assertFalse(gus.is_unweighted(self.A))

    def test_is_fully_connected(self):
        # graph where node at index [3] only connects to self
        A = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
        # fully connected graph
        B = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1], [0, 0, 0, 1]])
        self.assertFalse(gus.is_fully_connected(A))
        self.assertTrue(gus.is_fully_connected(B))

    def test_is_almost_symmetric(self):
        np.random.seed(8888)
        vec1 = np.random.normal(0, 1, (100, 100))
        vec2 = np.random.normal(0, 1, (100, 100))
        corr = np.corrcoef(vec1, vec2)
        self.assertTrue(gus.is_almost_symmetric(corr, atol=1e-15))
        self.assertFalse(gus.is_symmetric(corr))


class TestLCC(unittest.TestCase):
    def test_lcc_networkx(self):
        expected_lcc_matrix = np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )
        expected_nodelist = np.array([1, 2, 3, 4, 6])
        g = nx.DiGraph()
        [g.add_node(i) for i in range(1, 7)]
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        g.add_edge(3, 4)
        g.add_edge(3, 6)
        g.add_edge(6, 3)
        g.add_edge(4, 2)
        lcc, nodelist = gus.get_lcc(g, return_inds=True)
        lcc_matrix = nx.to_numpy_array(lcc)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)
        np.testing.assert_array_equal(nodelist, expected_nodelist)
        lcc = gus.get_lcc(g)
        lcc_matrix = nx.to_numpy_array(lcc)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)

    def test_lcc_networkx_undirected(self):
        expected_lcc_matrix = np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 1],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )
        expected_nodelist = np.array([1, 2, 3, 4, 6])
        g = nx.Graph()
        [g.add_node(i) for i in range(1, 7)]
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        g.add_edge(3, 6)
        g.add_edge(6, 3)
        g.add_edge(4, 2)
        lcc, nodelist = gus.get_lcc(g, return_inds=True)
        lcc_matrix = nx.to_numpy_array(lcc)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)
        np.testing.assert_array_equal(nodelist, expected_nodelist)
        lcc = gus.get_lcc(g)
        lcc_matrix = nx.to_numpy_array(lcc)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)

    def test_lcc_numpy(self):
        expected_lcc_matrix = np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )
        expected_nodelist = np.array([0, 1, 2, 3, 5])
        g = nx.DiGraph()
        [g.add_node(i) for i in range(1, 7)]
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        g.add_edge(3, 4)
        g.add_edge(3, 6)
        g.add_edge(6, 3)
        g.add_edge(4, 2)
        g = nx.to_numpy_array(g)
        lcc_matrix, nodelist = gus.get_lcc(g, return_inds=True)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)
        np.testing.assert_array_equal(nodelist, expected_nodelist)
        lcc_matrix = gus.get_lcc(g)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)

    def test_multigraph_lcc_numpystack(self):
        expected_g_matrix = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
        )
        expected_f_matrix = np.array(
            [[0, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
        )
        expected_mats = [expected_f_matrix, expected_g_matrix]
        expected_nodelist = np.array([0, 2, 3, 5])
        g = nx.DiGraph()
        [g.add_node(i) for i in range(1, 7)]
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        g.add_edge(3, 4)
        g.add_edge(3, 6)
        g.add_edge(6, 3)
        g.add_edge(4, 2)
        f = g.copy()
        f.add_edge(5, 4)
        f.remove_edge(4, 2)
        f.add_edge(3, 1)
        f = nx.to_numpy_array(f)
        g = nx.to_numpy_array(g)
        lccs, nodelist = gus.get_multigraph_intersect_lcc(
            np.stack([f, g]), return_inds=True
        )
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(graph, expected_mats[i])
            np.testing.assert_array_equal(nodelist, expected_nodelist)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(graph, expected_mats[i])

    def test_multigraph_lcc_recurse_numpylist(self):
        g = np.zeros((6, 6))
        g[0, 2] = 1
        g[2, 1] = 1
        g[1, 3] = 1
        g[3, 4] = 1
        # unconnected 5 for this graph

        f = np.zeros((6, 6))
        f[0, 1] = 1
        f[1, 3] = 1
        f[3, 4] = 1
        f[3, 5] = 1
        f[5, 2] = 1

        expected_g_lcc = expected_f_lcc = np.zeros((3, 3))
        expected_g_lcc[0, 1] = 1
        expected_g_lcc[1, 2] = 1
        expected_mats = [expected_g_lcc, expected_f_lcc]
        expected_nodelist = np.array([1, 3, 4])

        lccs, nodelist = gus.get_multigraph_intersect_lcc([f, g], return_inds=True)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(graph, expected_mats[i])
            np.testing.assert_array_equal(nodelist, expected_nodelist)

    def test_multigraph_lcc_numpylist(self):
        expected_g_matrix = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
        )
        expected_f_matrix = np.array(
            [[0, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
        )
        expected_mats = [expected_f_matrix, expected_g_matrix]
        expected_nodelist = np.array([0, 2, 3, 5])
        g = nx.DiGraph()
        [g.add_node(i) for i in range(1, 7)]
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        g.add_edge(3, 4)
        g.add_edge(3, 6)
        g.add_edge(6, 3)
        g.add_edge(4, 2)
        f = g.copy()
        f.add_edge(5, 4)
        f.remove_edge(4, 2)
        f.add_edge(3, 1)
        f = nx.to_numpy_array(f)
        g = nx.to_numpy_array(g)
        lccs, nodelist = gus.get_multigraph_intersect_lcc([f, g], return_inds=True)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(graph, expected_mats[i])
            np.testing.assert_array_equal(nodelist, expected_nodelist)
        lccs = gus.get_multigraph_intersect_lcc([f, g], return_inds=False)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(graph, expected_mats[i])

    def test_multigraph_lcc_networkx(self):
        expected_g_matrix = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
        )
        expected_f_matrix = np.array(
            [[0, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
        )
        expected_mats = [expected_f_matrix, expected_g_matrix]
        expected_nodelist = np.array([1, 3, 4, 6])
        g = nx.DiGraph()
        [g.add_node(i) for i in range(1, 7)]
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        g.add_edge(3, 4)
        g.add_edge(3, 6)
        g.add_edge(6, 3)
        g.add_edge(4, 2)
        f = g.copy()
        f.add_edge(5, 4)
        f.remove_edge(4, 2)
        f.add_edge(3, 1)
        lccs, nodelist = gus.get_multigraph_intersect_lcc([f, g], return_inds=True)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(nx.to_numpy_array(graph), expected_mats[i])
            np.testing.assert_array_equal(nodelist, expected_nodelist)
        lccs = gus.get_multigraph_intersect_lcc([f, g], return_inds=False)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(nx.to_numpy_array(graph), expected_mats[i])

    def test_multigraph_union(self):
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        B = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

        out_list = gus.get_multigraph_union_lcc([A, B])
        out_tensor = gus.get_multigraph_union_lcc(np.stack([A, B]))

        np.testing.assert_equal(out_list, [A, B])
        np.testing.assert_array_equal(out_tensor, np.stack([A, B]))


class TestDiagonalAugment(unittest.TestCase):
    def test_augment_diagonal_undirected(self):
        A = np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 0, 2, 1],
                [1, 0, 0, 1, 1],
                [0, 2, 1, 0, 0],
                [0, 1, 1, 0, 0],
            ]
        )
        expected = A.copy().astype(float)
        expected[0, 0] = 2.0 / 4
        expected[1, 1] = 4.0 / 4
        expected[2, 2] = 3.0 / 4
        expected[3, 3] = 3.0 / 4
        expected[4, 4] = 2.0 / 4
        A_aug = gus.augment_diagonal(A)
        np.testing.assert_array_equal(A_aug, expected)

    def test_augment_diagonal_directed(self):
        A = np.array(
            [
                [0, 1, -1, 0, 0],
                [0, 0, 0, 2, 1],
                [1, 0, 0, 1, 1],
                [0, 2, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )
        expected = A.copy().astype(float)
        expected[0, 0] = 1.5 / 4
        expected[1, 1] = 3 / 4
        expected[2, 2] = 2.5 / 4
        expected[3, 3] = 2.5 / 4
        expected[4, 4] = 1.5 / 4
        A_aug = gus.augment_diagonal(A)
        np.testing.assert_array_equal(A_aug, expected)

    def test_lcc_bad_matrix(self):
        A = np.array([0, 1])
        with self.assertRaises(ValueError):
            gus.get_lcc(A)


def test_binarize():
    g = np.array([[1, 2], [1, 1]])
    g2 = gus.binarize(g)
    g2_expected = np.ones_like(g)
    assert_equal(g2, g2_expected)
