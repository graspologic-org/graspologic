# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import random
import unittest
import warnings
from math import sqrt

import networkx as nx
import numpy as np
from numpy.testing import assert_equal
from scipy.sparse import csr_array

from graspologic.utils import remap_labels
from graspologic.utils import utils as gus


class TestAverageMatrices(unittest.TestCase):
    def test_mean_dense_and_sparse_are_equivalent(self):
        trials = 20

        for _ in range(trials):
            number_of_graphs = random.randint(2, 10)

            dim = random.randint(2, 100)
            dim2 = random.randint(2, 100)

            graphs = [np.random.rand(dim, dim2) for _ in range(number_of_graphs)]
            graphs_sparse = [csr_array(graph) for graph in graphs]

            graphs_averaged = gus.average_matrices(graphs)
            graphs_sparse_averaged = gus.average_matrices(graphs_sparse).todense()

            np.testing.assert_almost_equal(graphs_averaged, graphs_sparse_averaged)


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

    def test_to_laplacian_IDAD(self):
        expected_L_normed = [
            [1, -1 / (sqrt(2)), 0],
            [-1 / (sqrt(2)), 1, -1 / (sqrt(2))],
            [0, -1 / (sqrt(2)), 1],
        ]

        L_normed = gus.to_laplacian(self.A, form="I-DAD")
        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

    def test_to_laplacian_DAD(self):
        expected_L_normed = [
            [0, 1 / sqrt(2), 0],
            [1 / sqrt(2), 0, 1 / sqrt(2)],
            [0, 1 / sqrt(2), 0],
        ]

        L_normed = gus.to_laplacian(self.A, form="DAD")

        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

    def test_to_laplacian_RDAD(self):
        expected_L_normed = [
            [0, 3 / sqrt(70), 0],
            [3 / sqrt(70), 0, 3 / sqrt(70)],
            [0, 3 / sqrt(70), 0],
        ]

        L_normed = gus.to_laplacian(self.A, form="R-DAD")

        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

    def test_to_laplacian_regularizer_kwarg(self):
        expected_L_normed = [
            [0, 1 / sqrt(6), 0],
            [1 / sqrt(6), 0, 1 / sqrt(6)],
            [0, 1 / sqrt(6), 0],
        ]
        L_normed = gus.to_laplacian(self.A, form="R-DAD", regularizer=1.0)

        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

    def test_to_laplacian_symmetric(self):
        L_normed = gus.to_laplacian(self.A, form="DAD")

        self.assertTrue(gus.is_symmetric(L_normed))

    def test_to_laplacian_unsuported(self):
        with self.assertRaises(TypeError):
            gus.to_laplacian(self.A, form="MOM")

    def test_to_laplacian_unsuported_regularizer(self):
        with self.assertRaises(TypeError):
            gus.to_laplacian(self.A, form="R-DAD", regularizer="2")
        with self.assertRaises(TypeError):
            gus.to_laplacian(self.A, form="R-DAD", regularizer=[1, 2, 3])
        with self.assertRaises(ValueError):
            gus.to_laplacian(self.A, form="R-DAD", regularizer=-1.0)

    def test_to_laplacian_directed(self):
        expected_L_normed = [
            [0, 1 / 5, sqrt(5) / 10, 0.2],
            [0, 0, 0, sqrt(15) / 15],
            [0, sqrt(5) / 10, 0, sqrt(5) / 10],
            [0, sqrt(5) / 10, 0.25, 0],
        ]
        L_normed = gus.to_laplacian(self.B, form="R-DAD")
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

    def test_is_unweighted_for_mixed_zeros_ones_ndarray(self):
        B = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1.0, 0, 0], [1, 0, 1, 0]])
        self.assertTrue(gus.is_unweighted(B))

    def test_is_unweighted_for_random_ndarray(self):
        self.assertFalse(gus.is_unweighted(self.A))

    def test_is_unweighted_for_zeros_ndarray(self):
        self.assertTrue(gus.is_unweighted(np.zeros((10, 10))))

    def test_is_unweighted_for_ones_ndarray(self):
        self.assertTrue(gus.is_unweighted(np.ones((10, 10))))

    def test_is_unweighted_for_zeros_csr_array(self):
        m = csr_array((10, 10))
        self.assertTrue(gus.is_unweighted(m))

    def test_is_unweighted_for_ones_csr_array(self):
        m = csr_array(np.ones((10, 10)))
        self.assertTrue(gus.is_unweighted(m))

    def test_is_unweighted_for_mixed_zeros_ones_csr_array(self):
        m = csr_array((10_000, 10_000))
        m[0, 1] = 1
        m[1, 1] = 0
        m[9_999, 9_999] = 1.0
        self.assertTrue(gus.is_unweighted(m))

    def test_is_unweighted_for_random_csr_array(self):
        dim = 10_000
        num_nonzero = 20
        rows = np.random.randint(0, dim, num_nonzero)
        columns = np.random.randint(0, dim, num_nonzero)
        m = csr_array((dim, dim))
        m[rows, columns] = np.random.random(num_nonzero)
        self.assertFalse(gus.is_unweighted(m))

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
        expected_lcc_matrix = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ])
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
        lcc, nodelist = gus.largest_connected_component(g, return_inds=True)
        lcc_matrix = nx.to_numpy_array(lcc)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)
        np.testing.assert_array_equal(nodelist, expected_nodelist)
        lcc = gus.largest_connected_component(g)
        lcc_matrix = nx.to_numpy_array(lcc)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)

    def test_lcc_networkx_undirected(self):
        expected_lcc_matrix = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ])
        expected_nodelist = np.array([1, 2, 3, 4, 6])
        g = nx.Graph()
        [g.add_node(i) for i in range(1, 7)]
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        g.add_edge(3, 6)
        g.add_edge(6, 3)
        g.add_edge(4, 2)
        lcc, nodelist = gus.largest_connected_component(g, return_inds=True)
        lcc_matrix = nx.to_numpy_array(lcc)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)
        np.testing.assert_array_equal(nodelist, expected_nodelist)
        lcc = gus.largest_connected_component(g)
        lcc_matrix = nx.to_numpy_array(lcc)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)

    def test_lcc_numpy(self):
        expected_lcc_matrix = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ])
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
        lcc_matrix, nodelist = gus.largest_connected_component(g, return_inds=True)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)
        np.testing.assert_array_equal(nodelist, expected_nodelist)
        lcc_matrix = gus.largest_connected_component(g)
        np.testing.assert_array_equal(lcc_matrix, expected_lcc_matrix)

    def test_lcc_scipy(self):
        expected_lcc_matrix = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ])
        expected_nodelist = np.array([0, 1, 2, 3, 5])
        adjacency = np.array([
            [0, 1, 1, 0, 0, 0, 0],  # connected
            [0, 0, 0, 0, 0, 0, 0],  # connected
            [0, 0, 0, 1, 0, 1, 0],  # connected
            [0, 1, 0, 0, 0, 0, 0],  # connected
            [0, 0, 0, 0, 0, 0, 0],  # not connected
            [0, 0, 1, 0, 0, 0, 0],  # connected
            [0, 0, 0, 0, 0, 0, 0],  # not connected
        ])
        sparse_adjacency = csr_array(adjacency)

        lcc_matrix, nodelist = gus.largest_connected_component(
            sparse_adjacency, return_inds=True
        )
        np.testing.assert_array_equal(lcc_matrix.toarray(), expected_lcc_matrix)
        np.testing.assert_array_equal(nodelist, expected_nodelist)

    def test_lcc_scipy_empty(self):
        adjacency = np.array([[0, 1], [1, 0]])
        adjacency = csr_array(adjacency)

        # remove the actual connecting edges. this is now a disconnected graph
        # with two nodes. however, scipy still stores the entry that now has a 0 in it
        # as having a 'nonempty' value, which is used in the lcc calculation
        adjacency[0, 1] = 0
        adjacency[1, 0] = 0
        lcc_adjacency = gus.largest_connected_component(adjacency)
        assert lcc_adjacency.shape[0] == 1

    def test_multigraph_lcc_numpystack(self):
        expected_g_matrix = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        expected_f_matrix = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ])
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
        lccs, nodelist = gus.multigraph_lcc_intersection(
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

        lccs, nodelist = gus.multigraph_lcc_intersection([f, g], return_inds=True)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(graph, expected_mats[i])
            np.testing.assert_array_equal(nodelist, expected_nodelist)

    def test_multigraph_lcc_numpylist(self):
        expected_g_matrix = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        expected_f_matrix = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ])
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
        lccs, nodelist = gus.multigraph_lcc_intersection([f, g], return_inds=True)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(graph, expected_mats[i])
            np.testing.assert_array_equal(nodelist, expected_nodelist)
        lccs = gus.multigraph_lcc_intersection([f, g], return_inds=False)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(graph, expected_mats[i])

    def test_multigraph_lcc_networkx(self):
        expected_g_matrix = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        expected_f_matrix = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ])
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
        lccs, nodelist = gus.multigraph_lcc_intersection([f, g], return_inds=True)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(nx.to_numpy_array(graph), expected_mats[i])
            np.testing.assert_array_equal(nodelist, expected_nodelist)
        lccs = gus.multigraph_lcc_intersection([f, g], return_inds=False)
        for i, graph in enumerate(lccs):
            np.testing.assert_array_equal(nx.to_numpy_array(graph), expected_mats[i])

    def test_multigraph_union(self):
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        B = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

        out_list = gus.multigraph_lcc_union([A, B])
        out_tensor = gus.multigraph_lcc_union(np.stack([A, B]))

        np.testing.assert_equal(out_list, [A, B])
        np.testing.assert_array_equal(out_tensor, np.stack([A, B]))


class TestDiagonalAugment(unittest.TestCase):
    def test_augment_diagonal_undirected(self):
        A = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 0, 2, 1],
            [1, 0, 0, 1, 1],
            [0, 2, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ])
        expected = A.copy().astype(float)
        expected[0, 0] = 2.0 / 4
        expected[1, 1] = 4.0 / 4
        expected[2, 2] = 3.0 / 4
        expected[3, 3] = 3.0 / 4
        expected[4, 4] = 2.0 / 4
        A_aug = gus.augment_diagonal(A)
        np.testing.assert_array_equal(A_aug, expected)

    def test_augment_diagonal_directed(self):
        A = np.array([
            [0, 1, -1, 0, 0],
            [0, 0, 0, 2, 1],
            [1, 0, 0, 1, 1],
            [0, 2, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ])
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
            gus.largest_connected_component(A)


def test_binarize():
    g = np.array([[1, 2], [1, 1]])
    g2 = gus.binarize(g)
    g2_expected = np.ones_like(g)
    assert_equal(g2, g2_expected)


class TestRemoveVertices(unittest.TestCase):
    def setUp(self):
        self.directed = np.array([
            [0, 2, 3, 4, 5],
            [6, 0, 8, 9, 10],
            [11, 12, 0, 14, 15],
            [16, 17, 18, 0, 20],
            [21, 22, 23, 24, 0],
        ])
        self.undirected = np.array([
            [0, 6, 11, 16, 21],
            [6, 0, 12, 17, 22],
            [11, 12, 0, 18, 23],
            [16, 17, 18, 0, 24],
            [21, 22, 23, 24, 0],
        ])

    def test_undirected(self):
        # with list index
        indices = [0, -1, 1]
        for idx in [indices, np.array(indices)]:
            A, a = gus.remove_vertices(self.undirected, idx, return_removed=True)
            self.assertIsInstance(a, np.ndarray)
            assert_equal(A, np.array([[0, 18], [18, 0]]))
            assert_equal(a, np.array([[11, 16], [23, 24], [12, 17]]))
            self.assertTrue(gus.is_almost_symmetric(A))

        # with integer index
        indices = 0
        A, a = gus.remove_vertices(self.undirected, indices, return_removed=True)
        assert_equal(A, self.undirected[1:, 1:])
        assert_equal(a, np.array([6, 11, 16, 21]))
        self.assertTrue(gus.is_almost_symmetric(A))
        assert_equal(
            gus.remove_vertices(self.undirected, 0),
            gus.remove_vertices(self.undirected, [0]),
        )

    def test_directed(self):
        # with list index
        idx1 = [0, -1, 1]
        idx2 = np.array(idx1)
        for indices in [idx1, idx2]:
            A, a = gus.remove_vertices(self.directed, indices, return_removed=True)
            self.assertIsInstance(a, tuple)
            self.assertIsInstance(a[0], np.ndarray)
            self.assertIsInstance(a[1], np.ndarray)
            assert_equal(A, np.array([[0, 14], [18, 0]]))
            assert_equal(a[0], np.array([[11, 16], [15, 20], [12, 17]]))
            assert_equal(a[1], np.array([[3, 4], [23, 24], [8, 9]]))

        # with integer index
        idx = 0
        A, a = gus.remove_vertices(self.directed, idx, return_removed=True)
        assert_equal(A, gus.remove_vertices(self.directed, idx))
        self.assertIsInstance(a, tuple)
        self.assertIsInstance(a[0], np.ndarray)
        self.assertIsInstance(a[1], np.ndarray)
        assert_equal(A, self.directed[1:, 1:])
        assert_equal(a[0], np.array([6, 11, 16, 21]))
        assert_equal(a[1], np.array([2, 3, 4, 5]))

    def test_exceptions(self):
        # ensure proper errors are thrown when invalid inputs are passed.
        with self.assertRaises(TypeError):
            gus.remove_vertices(9001, 0)

        with self.assertRaises(ValueError):
            nonsquare = np.vstack((self.directed, self.directed))
            gus.remove_vertices(nonsquare, 0)

        with self.assertRaises(IndexError):
            indices = np.arange(len(self.directed) + 1)
            gus.remove_vertices(self.directed, indices)

        with self.assertRaises(IndexError):
            idx = len(self.directed) + 1
            gus.remove_vertices(self.directed, indices)


class TestRemapLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y_true = np.array([0, 0, 1, 1, 2, 2])
        cls.y_pred = np.array([1, 1, 2, 2, 0, 0])

    def test_proper_relabeling(self):
        y_pred_remapped, label_map = remap_labels(
            self.y_true, self.y_pred, return_map=True
        )
        assert_equal(y_pred_remapped, self.y_true)
        for key, val in label_map.items():
            if key == 0:
                assert val == 2
            elif key == 2:
                assert val == 1
            elif key == 1:
                assert val == 0

    def test_imperfect_relabeling(self):
        y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        # swap 0 and 1, make the 3rd label wrong.
        y_pred = [1, 1, 0, 0, 0, 0, 2, 2, 2]
        y_pred_remapped = remap_labels(y_true, y_pred)
        assert_equal(y_pred_remapped, [0, 0, 1, 1, 1, 1, 2, 2, 2])

    def test_string_relabeling(self):
        y_true = ["ant", "ant", "cat", "cat", "bird", "bird"]
        y_pred = ["bird", "bird", "cat", "cat", "ant", "ant"]
        y_pred_remapped = remap_labels(y_true, y_pred)
        assert_equal(y_true, y_pred_remapped)

    def test_inputs(self):
        with self.assertRaises(ValueError):
            # handled by sklearn confusion matrix
            remap_labels(self.y_true[1:], self.y_pred)

        with self.assertRaises(TypeError):
            remap_labels(8, self.y_pred)

        with self.assertRaises(TypeError):
            remap_labels(self.y_pred, self.y_true, return_map="hi")

        with self.assertRaises(ValueError):
            remap_labels(self.y_pred, ["ant", "ant", "cat", "cat", "bird", "bird"])


class TestRemapNodeIds(unittest.TestCase):
    def test_remap_node_ids_invalid_typ_raises_typeerror(self):
        invalid_types = [str, int, list]

        for type in invalid_types:
            with self.assertRaises(TypeError):
                gus.remap_node_ids(graph=type())

    def test_remap_node_ids_unweighted_graph_raises_warning(self):
        with warnings.catch_warnings(record=True) as warnings_context_manager:
            graph = nx.florentine_families_graph()

            gus.remap_node_ids(graph)

            self.assertEqual(len(warnings_context_manager), 1)
            self.assertTrue(
                issubclass(warnings_context_manager[0].category, UserWarning)
            )
            self.assertTrue(
                "Graph has at least one unweighted edge"
                in str(warnings_context_manager[0].message)
            )

    def _assert_graphs_are_equivalent(self, graph, new_graph, new_node_ids):
        self.assertTrue(len(new_graph.nodes()) == len(graph.nodes()))
        self.assertTrue(len(new_graph.edges()) == len(graph.edges()))

        for source, target, weight in graph.edges(data="weight"):
            self.assertTrue(source in new_node_ids.keys())
            self.assertTrue(target in new_node_ids.keys())

            new_weight = new_graph[new_node_ids[source]][new_node_ids[target]]["weight"]
            self.assertEqual(weight, new_weight)

    def test_remap_node_ids_graph_has_same_edges_but_remapped(self):
        graph = nx.Graph()

        graph.add_edge(0, 1, weight=10)
        graph.add_edge(1, "someid", weight=100)

        new_graph, new_node_ids = gus.remap_node_ids(graph)

        self._assert_graphs_are_equivalent(graph, new_graph, new_node_ids)

    def test_remap_node_ids_digraph_has_same_edges_but_remapped(self):
        graph = nx.DiGraph()

        graph.add_edge(0, 1, weight=10)
        graph.add_edge(1, "someid", weight=100)
        graph.add_edge("someid", 1, weight=21)

        new_graph, new_node_ids = gus.remap_node_ids(graph)

        self._assert_graphs_are_equivalent(graph, new_graph, new_node_ids)
