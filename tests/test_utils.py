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

    def test_ptr_loopless_undirected_zeroboost(self): 
        A = np.array([[0, 1, 0, 60], 
                      [1, 0, 400, 0],
                      [0, 400, 0, 80],
                      [60, 0, 80, 0]])
        ptr_expected = np.array([[0, 0.5, 0, 4.0/6.0],
                                 [0.5, 0, 1, 0],
                                 [0, 1, 0, 5.0/6],
                                 [4.0/6, 0, 5.0/6, 0]])
        
        ptr_out = gus.pass_to_ranks(A)
        self.assertTrue(np.allclose(ptr_out, ptr_expected, rtol=1e-04))

    def test_ptr_looped_undirected_zeroboost(self): 
        A = np.array([[0.5, 1, 0, 60], 
                        [1, 0, 400, 0],
                        [0, 400, 20, 80],
                        [60, 0, 80, 0]])
        ptr_expected = np.array([[0.5, 0.6, 0, 0.8],
                                    [0.6, 0, 1, 0],
                                    [0, 1, 0.7, 0.9],
                                    [0.8, 0, 0.9, 0]])
        
        ptr_out = gus.pass_to_ranks(A)
        self.assertTrue(np.allclose(ptr_out, ptr_expected, rtol=1e-04))

    def test_is_unweighted(self):
        B = np.array([[0, 1, 0, 0], 
                      [1, 0, 1, 0],
                      [0, 1.0, 0, 0],
                      [1, 0, 1, 0]])
        self.assertTrue(gus.is_unweighted(B))
        self.assertFalse(gus.is_unweighted(self.A))

    def test_ptr_invalid_inputs(self):
        with self.assertRaises(ValueError):
            gus.pass_to_ranks(self.A, method='hazelnutcoffe')
        with self.assertRaises(NotImplementedError):
            A = self.A
            A[2,0] = 1000 
            gus.pass_to_ranks(A)

if __name__ == '__main__':
    unittest.main()
