import unittest
import graphstats as gs
import numpy as np
import networkx as nx
from graphstats.simulations.simulations import er_nm


class TestSims(unittest.TestCase):

    def test_er(self):
        n = 10
        M = 20
        A = er_nm(n, M)
        # symmetric, so summing will give us twice the ecount of
        # the full adjacency matrix
        self.assertTrue(A.sum() == 2*M)
        self.assertTrue(A.shape == (n, n))