import unittest
import graspy as gs
import numpy as np
import networkx as nx
from graspy.utils import utils as gus


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

    def test_nonsquare(self):
        non_square = np.hstack((self.A, self.A))
        with self.assertRaises(ValueError):
            gus.import_graph(non_square)

if __name__ == '__main__':
    unittest.main()