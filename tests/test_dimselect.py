import unittest
import graphstats as gs
import numpy as np
import networkx as nx
from graphstats.simulations.simulations import *
from graphstats.utils.utils import is_symmetric, is_loopless
from graphstats.embed.dimselect import profile_likelihood

class Test_WSBM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 60 vertex graph w one community having 20 and another
        # w 40 vertices
        cls.n = [50, 70]
        cls.vcount = np.cumsum(cls.n)
        # define non-symmetric probability matrix as uneven
        cls.Pns = np.vstack(([0.6, 0.2], [0.3, 0.4]))
        # define symmetric probability as evenly weighted
        cls.Psy = np.vstack(([0.6, 0.2], [0.3, 0.4]))
        cls.Psy = symmetrize(cls.Psy)

    def test_binary_sbm(self):
        n = [50, 60, 70]
        vcount = np.cumsum(n)
        # define symmetric probability as evenly weighted
        Psy = np.vstack(([0.6, 0.2, 0.3], [0.3, 0.4, 0.2], [0.2, 0.8, 0.1]))
        Psy = symmetrize(Psy)
        np.random.seed(12345)
        A = binary_sbm(n, Psy)
        for i in range(0, len(n)):
            for j in range(0, len(n)):
                irange = np.arange(vcount[i] - n[i], vcount[i])
                jrange = np.arange(vcount[j] - n[j], vcount[j])

                block = A[(vcount[i] - n[i]):vcount[i],
                    (vcount[j] - n[j]):vcount[j]]
                if (i == j):
                    block = remove_diagonal(block)
                self.assertTrue(np.isclose(np.mean(block),
                    Psy[i, j], atol=0.02))
        self.assertTrue(is_symmetric(A))
        self.assertTrue(is_loopless(A))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(n), np.sum(n)))
        pass

    def test_weighted_sbm_singlewt_undirected_loopless(self):
        np.random.seed(12345)
        wt = np.random.normal
        params = {'loc': 2, 'scale': 2}
        A = weighted_sbm(self.n, self.Psy, Wt=wt, Wtargs=params)
        for i in range(0, len(self.n)):
            for j in range(0, len(self.n)):
                irange = np.arange(self.vcount[i] - self.n[i], self.vcount[i])
                jrange = np.arange(self.vcount[j] - self.n[j], self.vcount[j])

                block = A[(self.vcount[i] - self.n[i]):self.vcount[i],
                    (self.vcount[j] - self.n[j]):self.vcount[j]]
                if (i == j):
                    block = remove_diagonal(block)
                self.assertTrue(np.isclose(np.mean(block != 0),
                    self.Psy[i, j], atol=0.02))
                self.assertTrue(np.isclose(np.mean(block[block != 0]),
                    params['loc'], atol=0.2))
                self.assertTrue(np.isclose(np.std(block[block != 0]),
                    params['scale'], atol=0.2))
        self.assertTrue(is_symmetric(A))
        self.assertTrue(is_loopless(A))

if __name__ == '__main__':
    elbows, l, sings, all_l = profile_likelihood(data, 3)
