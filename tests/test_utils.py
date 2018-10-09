import unittest
import graspy as gs
import numpy as np
import networkx as nx
from graspy.utils import utils as gus
from math import sqrt

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
            np.random.normal(size=nedge))

    def test_graphin(self):
        G = nx.from_numpy_array(self.A)
        np.testing.assert_array_equal(
            nx.to_numpy_array(G), gus.import_graph(G))

    def test_npin(self):
        np.testing.assert_array_equal(self.A, gus.import_graph(self.A))

    def test_wrongtypein(self):
        a = 5
        with self.assertRaises(TypeError):
            gus.import_graph(a)
        with self.assertRaises(TypeError):
            gus.import_graph(None)

    def test_to_laplace_IDAD(self):
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        expected_L_normed = ([[1, -1 / (sqrt(2)),
                               0], [-1 / (sqrt(2)), 1, -1 / (sqrt(2))],
                              [0, -1 / (sqrt(2)), 1]])

        L_normed = gus.to_laplace(A, form='I-DAD')

        self.assertTrue(np.allclose(L_normed, expected_L_normed, rtol=1e-04))

    def test_to_laplace_unsuported(self):
        with self.assertRaises(TypeError):
            gus.to_laplace(self.A, form='MOM')

    def test_is_unweighted(self):
        B = np.array([[0, 1, 0, 0], 
                      [1, 0, 1, 0],
                      [0, 1.0, 0, 0],
                      [1, 0, 1, 0]])
        self.assertTrue(gus.is_unweighted(B))
        self.assertFalse(gus.is_unweighted(self.A))