import unittest
import graphstats as gs
import numpy as np
import networkx as nx
from graphstats.simulations.simulations import *
from graphstats.utils.utils import is_symmetric, is_loopless
import math


class TestER(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 15
        cls.M = 45
        cls.p = 0.2

    def test_ernm(self):
        A = er_nm(self.n, self.M)
        # symmetric, so summing will give us twice the ecount of
        # the full adjacency matrix
        self.assertTrue(A.sum() == 2*self.M)
        self.assertTrue(A.shape == (self.n, self.n))

    def test_ernp(self):
        np.random.seed(123456)
        A = er_np(self.n, self.p)
        # symmetric, so summing will give us twice the ecount of
        # the full adjacency matrix
        self.assertTrue(math.isclose(A.sum()/float(np.prod(A.shape)),
            self.p, abs_tol=0.02))
        self.assertTrue(A.shape == (self.n, self.n))


class TestZINM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 15
        cls.M = 40
        cls.wt = np.random.normal
        cls.mean = 2
        cls.std = 1

    def test_loop_directed(self):
        np.random.seed(12345)
        Abin = zi_nm(self.n, self.M, directed=True, loops=True)
        Awt = zi_nm(self.n, self.M, directed=True, loops=True, wt=self.wt,
            loc=self.mean, scale=self.std)
        # check that correct number of vertices assigned
        # sum of nonzero entries and correct for the fact that the diagonal
        # is part of the model now
        self.assertTrue(Abin.sum() == self.M)
        self.assertTrue((Awt != 0).sum() == self.M)


        # check that the nonzero vertices have mean self.mean and var self.var
        self.assertTrue(math.isclose(np.mean(Awt[Awt != 0]), self.mean,
            abs_tol=0.2))
        self.assertTrue(math.isclose(np.std(Awt[Awt != 0]), self.std,
            abs_tol=0.2))

        # check loopless and undirected
        self.assertFalse(is_symmetric(Abin))
        self.assertFalse(is_symmetric(Awt))
        self.assertFalse(is_loopless(Abin))
        self.assertFalse(is_loopless(Awt))

        # check dimensions
        self.assertTrue(Abin.shape == (self.n, self.n))
        self.assertTrue(Awt.shape == (self.n, self.n))
        pass

    def test_noloop_directed(self):
        np.random.seed(12345)
        Abin = zi_nm(self.n, self.M, directed=True)
        Awt = zi_nm(self.n, self.M, wt=self.wt, directed=True,
            loc=self.mean, scale=self.std)
        # check that correct number of vertices assigned
        self.assertTrue(Abin.sum() == self.M)
        self.assertTrue((Awt != 0).sum() == self.M)

        # check that the nonzero vertices have mean self.mean and var self.var
        self.assertTrue(math.isclose(np.mean(Awt[Awt != 0]), self.mean,
            abs_tol=0.2))
        self.assertTrue(math.isclose(np.std(Awt[Awt != 0]), self.std,
            abs_tol=0.2))

        # check loopless and undirected
        self.assertFalse(is_symmetric(Abin))
        self.assertFalse(is_symmetric(Awt))
        self.assertTrue(is_loopless(Abin))
        self.assertTrue(is_loopless(Awt))

        # check dimensions
        self.assertTrue(Abin.shape == (self.n, self.n))
        self.assertTrue(Awt.shape == (self.n, self.n))
        pass

    def test_loop_undirected(self):
        np.random.seed(12345)
        Abin = zi_nm(self.n, self.M, loops=True)
        Awt = zi_nm(self.n, self.M, loops=True, wt=self.wt,
            loc=self.mean, scale=self.std)
        # check that correct number of vertices assigned
        # sum of nonzero entries and correct for the fact that the diagonal
        # is part of the model now
        self.assertTrue(Abin.sum() + np.diag(Abin).sum() == 2*self.M)
        self.assertTrue((Awt != 0).sum() + np.diag(Awt != 0).sum() == 2*self.M)

        # check that the nonzero vertices have mean self.mean and var self.var
        self.assertTrue(math.isclose(np.mean(Awt[Awt != 0]), self.mean,
            abs_tol=0.2))
        self.assertTrue(math.isclose(np.std(Awt[Awt != 0]), self.std,
            abs_tol=0.2))

        # check loopless and undirected
        self.assertTrue(is_symmetric(Abin))
        self.assertTrue(is_symmetric(Awt))
        self.assertFalse(is_loopless(Abin))
        self.assertFalse(is_loopless(Awt))

        # check dimensions
        self.assertTrue(Abin.shape == (self.n, self.n))
        self.assertTrue(Awt.shape == (self.n, self.n))
        pass


    def test_noloop_undirected(self):
        np.random.seed(12345)
        Abin = zi_nm(self.n, self.M)
        Awt = zi_nm(self.n, self.M, wt=self.wt, loc=self.mean, scale=self.std)
        # check that correct number of vertices assigned
        self.assertTrue(Abin.sum() == 2*self.M)
        self.assertTrue((Awt != 0).sum() == 2*self.M)

        # check that the nonzero vertices have mean self.mean and var self.var
        self.assertTrue(math.isclose(np.mean(Awt[Awt != 0]), self.mean,
            abs_tol=0.2))
        self.assertTrue(math.isclose(np.std(Awt[Awt != 0]), self.std,
            abs_tol=0.2))

        # check loopless and undirected
        self.assertTrue(is_symmetric(Abin))
        self.assertTrue(is_symmetric(Awt))
        self.assertTrue(is_loopless(Abin))
        self.assertTrue(is_loopless(Awt))

        # check dimensions
        self.assertTrue(Abin.shape == (self.n, self.n))
        self.assertTrue(Awt.shape == (self.n, self.n))
        pass


class TestZINP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 15
        cls.p = 0.2
        cls.wt = np.random.normal
        cls.mean = 1
        cls.var = 1

    def test_loop_directed(self):
        pass

    def test_noloop_directed(self):
        pass

    def test_loop_undirected(self):
        pass

    def test_noloop_undirected(self):
        pass