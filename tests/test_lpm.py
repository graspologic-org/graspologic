import unittest
import graspy as gs
import numpy as np
import networkx as nx
from graspy.embed.lpm import LatentPosition


class TestLPM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 10
        d = 5
        cls.X = np.random.rand(n, d).reshape((n, d))
        cls.Y = np.random.rand(n, d).reshape((n, d))

    def test_symmetric(self):
        lpm = LatentPosition(self.X, self.X)
        self.assertTrue(lpm.is_symmetric())
        self.assertTrue(lpm.Y is None)
        np.testing.assert_equal(lpm.X, self.X)

    def test_asymmetric(self):
        lpm = LatentPosition(self.X, self.Y)
        self.assertFalse(lpm.is_symmetric())
        self.assertFalse(lpm.Y is None)
        np.testing.assert_equal(lpm.X, self.X)
        np.testing.assert_equal(lpm.Y, self.Y)
