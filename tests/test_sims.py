import unittest
import graspy as gs
import numpy as np
import networkx as nx
from graspy.simulations import *
from graspy.utils.utils import is_symmetric, is_loopless


def remove_diagonal(A):
    # indices of A
    Aind = np.ravel_multi_index(np.indices(A.shape), A.shape)
    # indices of the diagonal
    eind = np.ravel_multi_index(np.where(np.eye(A.shape[1])), A.shape)
    # set difference of A indices and identity
    dind = np.unravel_index(np.setdiff1d(Aind, eind), A.shape)
    return A[dind]


class Test_ER(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 20
        cls.M = 45
        cls.p = 0.2

    def test_ernm(self):
        A = er_nm(self.n, self.M)
        # symmetric, so summing will give us twice the ecount of
        # the full adjacency matrix
        self.assertTrue(A.sum() == 2 * self.M)
        self.assertTrue(A.shape == (self.n, self.n))

    def test_ernp(self):
        np.random.seed(123456)
        A = er_np(self.n, self.p)
        # symmetric, so summing will give us twice the ecount of
        # the full adjacency matrix
        dind = remove_diagonal(A)
        self.assertTrue(np.isclose(dind.sum() / float(len(dind)), self.p, atol=0.02))
        self.assertTrue(A.shape == (self.n, self.n))


class Test_ZINM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 20
        cls.M = 120
        cls.wt = np.random.normal
        cls.mean = 2
        cls.std = 1
        cls.wtargs = dict(loc=cls.mean, scale=cls.std)
        cls.seed = 12345

    def test_loop_directed(self):
        np.random.seed(12345)
        Abin = er_nm(self.n, self.M, directed=True, loops=True)
        Awt = er_nm(
            self.n, self.M, directed=True, loops=True, wt=self.wt, wtargs=self.wtargs
        )
        # check that correct number of edges assigned
        # sum of nonzero entries and correct for the fact that the diagonal
        # is part of the model now
        self.assertTrue(Abin.sum() == self.M)
        self.assertTrue((Awt != 0).sum() == self.M)

        # check that the nonzero edges have mean self.mean and var self.var
        self.assertTrue(np.isclose(np.mean(Awt[Awt != 0]), self.mean, atol=0.2))
        self.assertTrue(np.isclose(np.std(Awt[Awt != 0]), self.std, atol=0.2))

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
        Abin = er_nm(self.n, self.M, directed=True)
        Awt = er_nm(self.n, self.M, wt=self.wt, directed=True, wtargs=self.wtargs)
        # check that correct number of edges assigned
        self.assertTrue(Abin.sum() == self.M)
        self.assertTrue((Awt != 0).sum() == self.M)

        dind = remove_diagonal(Awt)
        # check that the nonzero edges have mean self.mean and var self.var
        self.assertTrue(np.isclose(np.mean(dind[dind != 0]), self.mean, atol=0.15))
        self.assertTrue(np.isclose(np.std(dind[dind != 0]), self.std, atol=0.15))

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
        Abin = er_nm(self.n, self.M, loops=True)
        Awt = er_nm(self.n, self.M, loops=True, wt=self.wt, wtargs=self.wtargs)
        # check that correct number of edges assigned
        # sum of nonzero entries and correct for the fact that the diagonal
        # is part of the model now
        self.assertTrue(Abin.sum() + np.diag(Abin).sum() == 2 * self.M)
        self.assertTrue((Awt != 0).sum() + np.diag(Awt != 0).sum() == 2 * self.M)

        # check that the nonzero edges have mean self.mean and var self.var
        self.assertTrue(np.isclose(np.mean(Awt[Awt != 0]), self.mean, atol=0.15))
        self.assertTrue(np.isclose(np.std(Awt[Awt != 0]), self.std, atol=0.15))

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
        Abin = er_nm(self.n, self.M)
        Awt = er_nm(self.n, self.M, wt=self.wt, wtargs=self.wtargs)
        # check that correct number of edges assigned
        self.assertTrue(Abin.sum() == 2 * self.M)
        self.assertTrue((Awt != 0).sum() == 2 * self.M)

        dind = remove_diagonal(Awt)
        # check that the nonzero edges have mean self.mean and var self.var
        self.assertTrue(np.isclose(np.mean(dind[dind != 0]), self.mean, atol=0.15))
        self.assertTrue(np.isclose(np.std(dind[dind != 0]), self.std, atol=0.15))

        # check loopless and undirected
        self.assertTrue(is_symmetric(Abin))
        self.assertTrue(is_symmetric(Awt))
        self.assertTrue(is_loopless(Abin))
        self.assertTrue(is_loopless(Awt))

        # check dimensions
        self.assertTrue(Abin.shape == (self.n, self.n))
        self.assertTrue(Awt.shape == (self.n, self.n))

    def test_bad_inputs(self):
        with self.assertRaises(TypeError):
            n = "10"
            er_nm(n, self.M)

        with self.assertRaises(ValueError):
            n = -1
            er_nm(n, self.M)

        with self.assertRaises(TypeError):
            m = 1.0
            er_nm(self.n, m)

        with self.assertRaises(ValueError):
            m = -1
            er_nm(self.n, m)

        with self.assertRaises(TypeError):
            loops = "True"
            er_nm(self.n, self.M, loops=loops)

        with self.assertRaises(TypeError):
            directed = "True"
            er_nm(self.n, self.M, directed=directed)

        with self.assertRaises(TypeError):
            wt = np.random
            er_nm(self.n, self.M, wt=wt)

        with self.assertRaises(ValueError):
            m = 10000
            er_nm(self.n, m)


class Test_ZINP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 50
        cls.p = 0.5
        cls.wt = np.random.normal
        cls.mean = 2
        cls.std = 1
        cls.wtargs = dict(loc=cls.mean, scale=cls.std)
        cls.seed = 123

    def test_loop_directed(self):
        np.random.seed(123)
        Abin = er_np(self.n, self.p, directed=True, loops=True)
        Awt = er_np(
            self.n, self.p, directed=True, loops=True, wt=self.wt, wtargs=self.wtargs
        )
        # check that correct number of edges assigned
        # sum of nonzero entries and correct for the fact that the diagonal
        # is part of the model now
        self.assertTrue(
            np.isclose(Abin.sum() / float(np.prod(Abin.shape)), self.p, atol=0.1)
        )
        self.assertTrue(
            np.isclose((Awt != 0).sum() / float(np.prod(Awt.shape)), self.p, atol=0.1)
        )

        # check that the nonzero edges have mean self.mean and var self.var
        self.assertTrue(np.isclose(np.mean(Awt[Awt != 0]), self.mean, atol=0.2))
        self.assertTrue(np.isclose(np.std(Awt[Awt != 0]), self.std, atol=0.2))

        # check loopless and undirected
        self.assertFalse(is_symmetric(Abin))
        self.assertFalse(is_symmetric(Awt))
        self.assertFalse(is_loopless(Abin))
        self.assertFalse(is_loopless(Awt))

        # check dimensions
        self.assertTrue(Abin.shape == (self.n, self.n))
        self.assertTrue(Awt.shape == (self.n, self.n))

    def test_noloop_directed(self):
        np.random.seed(12345)
        Abin = er_np(self.n, self.p, directed=True)
        Awt = er_np(self.n, self.p, wt=self.wt, directed=True, wtargs=self.wtargs)
        # check that correct number of edges assigned
        dind = remove_diagonal(Abin)
        dindwt = remove_diagonal(Awt)
        self.assertTrue(np.isclose(dind.sum() / float(len(dind)), self.p, atol=0.1))
        self.assertTrue(
            np.isclose((dindwt != 0).sum() / float(len(dindwt)), self.p, atol=0.1)
        )

        # check that the nonzero edges have mean self.mean and var self.var
        self.assertTrue(np.isclose(np.mean(dindwt[dindwt != 0]), self.mean, atol=0.5))
        self.assertTrue(np.isclose(np.std(dindwt[dindwt != 0]), self.std, atol=0.5))

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
        Abin = er_np(self.n, self.p, loops=True)
        Awt = er_np(self.n, self.p, loops=True, wt=self.wt, wtargs=self.wtargs)
        # check that correct number of edges assigned
        self.assertTrue(
            np.isclose(Abin.sum() / float(np.prod(Abin.shape)), self.p, atol=0.02)
        )
        self.assertTrue(
            np.isclose((Awt != 0).sum() / float(np.prod(Awt.shape)), self.p, atol=0.02)
        )
        # check that the nonzero edges have mean self.mean and var self.var
        self.assertTrue(np.isclose(np.mean(Awt[Awt != 0]), self.mean, atol=0.15))
        self.assertTrue(np.isclose(np.std(Awt[Awt != 0]), self.std, atol=0.15))

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
        np.random.seed(123)
        Abin = er_np(self.n, self.p)
        Awt = er_np(self.n, self.p, wt=self.wt, wtargs=self.wtargs)
        # check that correct number of edges assigned
        dind = remove_diagonal(Abin)
        dindwt = remove_diagonal(Awt)
        self.assertTrue(np.isclose(dind.sum() / float(len(dind)), self.p, atol=0.02))
        self.assertTrue(
            np.isclose((dindwt != 0).sum() / float(len(dindwt)), self.p, atol=0.02)
        )

        # check that the nonzero edges have mean self.mean and var self.var
        self.assertTrue(np.isclose(np.mean(dindwt[dindwt != 0]), self.mean, atol=0.15))
        self.assertTrue(np.isclose(np.std(dindwt[dindwt != 0]), self.std, atol=0.15))

        # check loopless and undirected
        self.assertTrue(is_symmetric(Abin))
        self.assertTrue(is_symmetric(Awt))
        self.assertTrue(is_loopless(Abin))
        self.assertTrue(is_loopless(Awt))

        # check dimensions
        self.assertTrue(Abin.shape == (self.n, self.n))
        self.assertTrue(Awt.shape == (self.n, self.n))

    def test_bad_inputs(self):
        with self.assertRaises(TypeError):
            n = "10"
            er_np(n, self.p)

        with self.assertRaises(ValueError):
            n = -1
            er_np(n, self.p)

        with self.assertRaises(TypeError):
            p = "1"
            er_np(self.n, p)

        with self.assertRaises(ValueError):
            p = -0.5
            er_np(self.n, p)

        with self.assertRaises(ValueError):
            p = 5.0
            er_np(self.n, p)

        with self.assertRaises(TypeError):
            loops = "True"
            er_np(self.n, self.p, loops=loops)

        with self.assertRaises(TypeError):
            directed = "True"
            er_np(self.n, self.p, directed=directed)

        with self.assertRaises(TypeError):
            wt = np.random
            er_np(self.n, self.p, wt=wt)

        with self.assertRaises(TypeError):
            dc = np.array(np.random.power)
            er_np(self.n, self.p, dc=dc)

def modular_edges(n):
    """
    A function for generating modular sbm edge communities.
    """
    m = int(n/2)
    edge_comm = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if ( (i<m) & (j<m)) or ( (i>=m ) & (j>=m) ):
                edge_comm[i,j] = 1
            else:
                edge_comm[i,j] = 2
    return edge_comm

def diag_edges(n):
    """
    A function for generating diagonal SIEM edge communities.
    """
    m = int(n/2)
    edge_comm = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (i == j + m) or (j == i + m):
                edge_comm[i,j] = 1
            else:
                edge_comm[i,j] = 2
    return edge_comm

class Test_SIEM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 100  # number of vertices
        # loopy symmetric edge communities
        cls.Esy_loop = modular_edges(cls.n)
        # loopy asymmetric edge communities
        cls.Ens_loop = np.copy(cls.Esy_loop)
        cls.Ens_loop[2, 1] = 2
        # loopless symmetric
        cls.Esy_ll = cls.Esy_loop - np.diag(np.diag(cls.Esy_loop))
        # loopless asymmetric
        cls.Esy_ll = cls.Esy_loop - np.diag(np.diag(cls.Ens_loop))
        cls.Ediag = diag_edges(cls.n)

    def test_siem_label(self):
        np.random.seed(1)
        A, E = siem(self.n, 1, self.Ens_loop, return_labels=True, loops=True)
        self.assertTrue(np.allclose(E, self.Ens_loop))
        pass

    def test_uw_siem_loops(self):
        np.random.seed(1)
        A = siem(self.n, 1, self.Esy_loop, loops=True)
        self.assertTrue(A.sum() == 100**2)  # test that matrix is all 1s
        A = siem(self.n, [1, .5], self.Esy_loop, loops=True)
        self.assertTrue(A[self.Esy_loop == 1].sum() == (100**2)/2)
        self.assertTrue(np.allclose(A[self.Esy_loop == 2].mean(), 0.5, atol=.01))
        pass

    def test_bad_inputs(self):
        # n is not passed as an integer
        with self.assertRaises(TypeError):
            n = "1"
            siem(n, 1, self.Ens_loop)
        # loops is not a boolean
        with self.assertRaises(TypeError):
            n = "1"
            siem(n, 1, self.Ens_loop, loops="maybe")

        # directed is not a boolean
        with self.assertRaises(TypeError):
            siem(self.n, 1, self.Ens_loop, directed="yes")

        # edge_comm contains non-numeric entries
        with self.assertRaises(ValueError):
            E = np.array([["yes", "no"], ["q", "a"]])
            siem(2, 1, E)

        # edge_comm contains numeric, but non-integer entries
        with self.assertRaises(ValueError):
            E = np.copy(self.Ens_loop)
            E[1,1] = 1.111
            siem(self.n, 1, E, loops=True)


        # edge_comm contains negative entries
        with self.assertRaises(ValueError):
            E = np.copy(self.Ens_loop)
            E[1,1] = -1
            siem(self.n, 1, E, loops=True)

        # edge_comm is not a 2d array or numpy matrix
        with self.assertRaises(ValueError):
            E = np.dstack((self.Ens_loop, self.Ens_loop))
            siem(self.n, 1, E, loops=True)

        # edge_comm does not contain entries 1:K

        # edge_comm has a minimum above 1 when loopy

        # edge_comm has a minimum above 1 when loopless in off-diagonals

        # have list p but too many entries
        with self.assertRaises(ValueError):
            siem(self.n, [1,1,1], self.Ens_loop, loops=True)

        # have list p but too few entries
        with self.assertRaises(ValueError):
            E = np.copy(self.Ens_loop)
            E[1,1] = 3
            siem(self.n, [1,1], E, loops=True)

        # pass in p with non-numerics
        with self.assertRaises(ValueError):
            siem(self.n, ["yes", 1, 1], self.Ens_loop, loops=True)

        # pass in p with non-probability floats
        with self.assertRaises(ValueError):
            siem(self.n, [1.5, 1, 1], self.Ens_loop, loops=True)

        # pass in p that is not a list or vector
        with self.assertRaises(ValueError):
            siem(self.n, np.ones((2, 2)), self.Ens_loop, loops=True)

        # request loopless, but pass loopy edge communities
        with self.assertRaises(ValueError):
            siem(self.n, [1,1], self.Ens_loop, loops=False)

        # request loopy, but pass in loopless edge communities
        with self.assertRaises(ValueError):
            siem(self.n, [1,1], self.Esy_ll, loops=True, directed=False)

        # request undirected, but pass in asymmetric edge communities
        with self.assertRaises(ValueError):
            siem(self.n, [1,1], self.Esy_ll, directed=False, loops=True)

        # wt is not callable
        with self.assertRaises(TypeError):
            siem(self.n, [1,1], self.Esy_ll, loops=False, directed=False, wt="test", wtargs={})

        # wt is a list, but wtargs is None
        with self.assertRaises(TypeError):
            siem(self.n, [1,1], self.Esy_ll, loops=False, directed=False, wt=np.random.normal, wtargs=None)

        # wt is a list, but wtargs is not
        with self.assertRaises(TypeError):
            siem(self.n, [1,1], self.Esy_ll, loops=False, directed=False,
                wt=[np.random.normal, np.random.normal], wtargs={"loc": 0.0, "scale":"1.0"})

        # wtargs is not a dictionary
        with self.assertRaises(TypeError):
            siem(self.n, [1,1], self.Esy_ll, loops=False, directed=False,
                wt=[np.random.normal, np.random.normal], wtargs="test2")

        # wtargs is a list of non-dictionaries
        with self.assertRaises(TypeError):
            siem(self.n, [1,1], self.Esy_ll, loops=False, directed=False,
                wt=[np.random.normal, np.random.normal], wtargs=[0.0, 1.0])
        
        # too few wts

        # too many wts
        pass



class Test_WSBM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 120 vertex graph w one community having 50 and another
        # w 70 vertices
        cls.n = [50, 70]
        cls.vcount = np.cumsum(cls.n)
        # define non-symmetric probability matrix as uneven
        cls.Pns = np.vstack(([0.6, 0.2], [0.3, 0.4]))
        # define symmetric probability as evenly weighted
        cls.Psy = np.vstack(([0.6, 0.2], [0.3, 0.4]))
        cls.Psy = symmetrize(cls.Psy)
        cls.seed = 12345

    def test_sbm_label(self):
        np.random.seed(1)
        n = [3, 3]
        p = [[0.5, 0.1], [0.1, 0.5]]
        A, l = sbm(n, p, return_labels=True)
        label = [0, 0, 0, 1, 1, 1]
        self.assertTrue(np.allclose(l, label))

    def test_sbm(self):
        n = [50, 60, 70]
        vcount = np.cumsum(n)
        # define symmetric probability as evenly weighted
        Psy = np.vstack(([0.6, 0.2, 0.3], [0.3, 0.4, 0.2], [0.2, 0.8, 0.1]))
        Psy = symmetrize(Psy)
        np.random.seed(12345)
        A = sbm(n, Psy)
        for i in range(0, len(n)):
            for j in range(0, len(n)):
                irange = np.arange(vcount[i] - n[i], vcount[i])
                jrange = np.arange(vcount[j] - n[j], vcount[j])

                block = A[
                    (vcount[i] - n[i]) : vcount[i], (vcount[j] - n[j]) : vcount[j]
                ]
                if i == j:
                    block = remove_diagonal(block)
                self.assertTrue(np.isclose(np.mean(block), Psy[i, j], atol=0.02))
        self.assertTrue(is_symmetric(A))
        self.assertTrue(is_loopless(A))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(n), np.sum(n)))
        pass

    def test_sbm_singlewt_undirected_loopless(self):
        np.random.seed(12345)
        wt = np.random.normal
        params = {"loc": 2, "scale": 2}
        A = sbm(self.n, self.Psy, wt=wt, wtargs=params)
        for i in range(0, len(self.n)):
            for j in range(0, len(self.n)):
                irange = np.arange(self.vcount[i] - self.n[i], self.vcount[i])
                jrange = np.arange(self.vcount[j] - self.n[j], self.vcount[j])

                block = A[
                    (self.vcount[i] - self.n[i]) : self.vcount[i],
                    (self.vcount[j] - self.n[j]) : self.vcount[j],
                ]
                if i == j:
                    block = remove_diagonal(block)
                self.assertTrue(
                    np.isclose(np.mean(block != 0), self.Psy[i, j], atol=0.02)
                )
                self.assertTrue(
                    np.isclose(np.mean(block[block != 0]), params["loc"], atol=0.2)
                )
                self.assertTrue(
                    np.isclose(np.std(block[block != 0]), params["scale"], atol=0.2)
                )
        self.assertTrue(is_symmetric(A))
        self.assertTrue(is_loopless(A))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(self.n), np.sum(self.n)))

    # below are the expectations of the estimators for the relevant weight
    # functions we exhaustively test
    def exp_normal(self, x):
        return {"loc": np.mean(x), "scale": np.std(x)}

    def exp_poisson(self, x):
        return {"lam": np.mean(x)}

    def exp_exp(self, x):
        return {"scale": np.mean(x)}

    def exp_unif(self, x):
        return {"low": np.min(x), "high": np.max(x)}

    def test_sbm_multiwt_directed_loopless(self):
        np.random.seed(12345)
        Wt = np.vstack(
            (
                [np.random.normal, np.random.poisson],
                [np.random.exponential, np.random.uniform],
            )
        )
        Wtargs = np.vstack(
            (
                [{"loc": 2, "scale": 2}, {"lam": 5}],
                [{"scale": 2}, {"low": 5, "high": 10}],
            )
        )
        check = np.vstack(
            ([self.exp_normal, self.exp_poisson], [self.exp_exp, self.exp_unif])
        )
        A = sbm(self.n, self.Psy, wt=Wt, directed=True, wtargs=Wtargs)
        for i in range(0, len(self.n)):
            for j in range(0, len(self.n)):
                irange = np.arange(self.vcount[i] - self.n[i], self.vcount[i])
                jrange = np.arange(self.vcount[j] - self.n[j], self.vcount[j])

                block = A[
                    (self.vcount[i] - self.n[i]) : self.vcount[i],
                    (self.vcount[j] - self.n[j]) : self.vcount[j],
                ]
                if i == j:
                    block = remove_diagonal(block)
                self.assertTrue(
                    np.isclose(np.mean(block != 0), self.Psy[i, j], atol=0.02)
                )
                fit = check[i, j](block[block != 0])
                for k, v in fit.items():
                    self.assertTrue(np.isclose(v, Wtargs[i, j][k], atol=0.2))
        self.assertFalse(is_symmetric(A))
        self.assertTrue(is_loopless(A))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(self.n), np.sum(self.n)))
        pass

    def test_sbm_multiwt_undirected_loopless(self):
        np.random.seed(12345)
        Wt = np.vstack(
            (
                [np.random.normal, np.random.poisson],
                [np.random.poisson, np.random.uniform],
            )
        )
        Wtargs = np.vstack(
            ([{"loc": 2, "scale": 2}, {"lam": 5}], [{"lam": 5}, {"low": 5, "high": 10}])
        )
        check = np.vstack(
            ([self.exp_normal, self.exp_poisson], [self.exp_poisson, self.exp_unif])
        )
        A = sbm(self.n, self.Psy, wt=Wt, directed=False, wtargs=Wtargs)
        for i in range(0, len(self.n)):
            for j in range(0, len(self.n)):
                irange = np.arange(self.vcount[i] - self.n[i], self.vcount[i])
                jrange = np.arange(self.vcount[j] - self.n[j], self.vcount[j])

                block = A[
                    (self.vcount[i] - self.n[i]) : self.vcount[i],
                    (self.vcount[j] - self.n[j]) : self.vcount[j],
                ]
                if i == j:
                    block = remove_diagonal(block)
                self.assertTrue(
                    np.isclose(np.mean(block != 0), self.Psy[i, j], atol=0.02)
                )
                fit = check[i, j](block[block != 0])
                for k, v in fit.items():
                    self.assertTrue(np.isclose(v, Wtargs[i, j][k], atol=0.2))
        self.assertTrue(is_symmetric(A))
        self.assertTrue(is_loopless(A))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(self.n), np.sum(self.n)))
        pass

    def test_sbm_multiwt_directed_loopy(self):
        np.random.seed(12345)
        Wt = np.vstack(
            (
                [np.random.normal, np.random.poisson],
                [np.random.exponential, np.random.uniform],
            )
        )
        Wtargs = np.vstack(
            (
                [{"loc": 2, "scale": 2}, {"lam": 5}],
                [{"scale": 2}, {"low": 5, "high": 10}],
            )
        )
        check = np.vstack(
            ([self.exp_normal, self.exp_poisson], [self.exp_exp, self.exp_unif])
        )
        A = sbm(self.n, self.Psy, wt=Wt, directed=True, loops=True, wtargs=Wtargs)
        for i in range(0, len(self.n)):
            for j in range(0, len(self.n)):
                irange = np.arange(self.vcount[i] - self.n[i], self.vcount[i])
                jrange = np.arange(self.vcount[j] - self.n[j], self.vcount[j])

                block = A[
                    (self.vcount[i] - self.n[i]) : self.vcount[i],
                    (self.vcount[j] - self.n[j]) : self.vcount[j],
                ]
                self.assertTrue(
                    np.isclose(np.mean(block != 0), self.Psy[i, j], atol=0.02)
                )
                fit = check[i, j](block[block != 0])
                for k, v in fit.items():
                    self.assertTrue(np.isclose(v, Wtargs[i, j][k], atol=0.2))
        self.assertFalse(is_symmetric(A))
        self.assertFalse(is_loopless(A))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(self.n), np.sum(self.n)))
        pass

    def test_sbm_multiwt_undirected_loopy(self):
        np.random.seed(12345)
        Wt = np.vstack(
            (
                [np.random.normal, np.random.poisson],
                [np.random.poisson, np.random.uniform],
            )
        )
        Wtargs = np.vstack(
            ([{"loc": 2, "scale": 2}, {"lam": 5}], [{"lam": 5}, {"low": 5, "high": 10}])
        )
        check = np.vstack(
            ([self.exp_normal, self.exp_poisson], [self.exp_poisson, self.exp_unif])
        )
        A = sbm(self.n, self.Psy, wt=Wt, directed=False, loops=True, wtargs=Wtargs)
        for i in range(0, len(self.n)):
            for j in range(0, len(self.n)):
                irange = np.arange(self.vcount[i] - self.n[i], self.vcount[i])
                jrange = np.arange(self.vcount[j] - self.n[j], self.vcount[j])

                block = A[
                    (self.vcount[i] - self.n[i]) : self.vcount[i],
                    (self.vcount[j] - self.n[j]) : self.vcount[j],
                ]
                self.assertTrue(
                    np.isclose(np.mean(block != 0), self.Psy[i, j], atol=0.02)
                )
                fit = check[i, j](block[block != 0])
                for k, v in fit.items():
                    self.assertTrue(np.isclose(v, Wtargs[i, j][k], atol=0.2))
        self.assertTrue(is_symmetric(A))
        self.assertFalse(is_loopless(A))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(self.n), np.sum(self.n)))
        pass

    def test_sbm_dc_dc_kws_directed_loopy_weights(self):
        np.random.seed(self.seed)
        funcs = [np.random.power, np.random.uniform]
        dc_kwss = [{"a": 3}, {"low": 5, "high": 10}]
        dc = np.hstack(
            (
                [
                    [funcs[i](**dc_kwss[i]) for _ in range(self.n[i])]
                    for i in range(len(self.n))
                ]
            )
        )
        for i in range(0, len(self.n)):
            dc[self.vcount[i] - self.n[i] : self.vcount[i]] /= sum(
                dc[self.vcount[i] - self.n[i] : self.vcount[i]]
            )
        A = sbm(self.n, self.Psy, directed=True, loops=True, dc=dc)
        communities = np.hstack([[comm] * self.n[comm] for comm in range(len(self.n))])
        for i, ki in zip(range(sum(self.n)), communities):
            degree = sum([A[i][j] for j in range(sum(self.n))])
            theta_hat = degree / sum(
                [
                    self.Psy[ki][kj] * self.n[ki] * self.n[kj]
                    for kj in range(len(self.n))
                ]
            )
            self.assertTrue(np.isclose(theta_hat, dc[i], atol=0.01))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(self.n), np.sum(self.n)))
        pass

    def test_sbm_dc_dc_kws_directed_loopy(self):
        np.random.seed(self.seed)
        funcs = [np.random.power, np.random.uniform]
        dc_kwss = [{"a": 3}, {"low": 5, "high": 10}]
        for i in range(len(funcs)):
            A = sbm(
                self.n,
                self.Psy,
                directed=True,
                loops=True,
                dc=funcs[i],
                dc_kws=dc_kwss[i],
            )
            for i in range(0, len(self.n)):
                for j in range(0, len(self.n)):
                    block = A[
                        (self.vcount[i] - self.n[i]) : self.vcount[i],
                        (self.vcount[j] - self.n[j]) : self.vcount[j],
                    ]
                    if i == j:
                        block = remove_diagonal(block)
                    self.assertTrue(
                        np.isclose(np.mean(block), self.Psy[i, j], atol=0.02)
                    )
            self.assertFalse(is_symmetric(A))
            self.assertFalse(is_loopless(A))
            # check dimensions
            self.assertTrue(A.shape == (np.sum(self.n), np.sum(self.n)))
        pass

    def test_sbm_multi_dc_dc_kws(self):
        np.random.seed(self.seed)
        dc = [np.random.power, np.random.uniform]
        dc_kws = [{"a": 3}, {"low": 5, "high": 10}]
        A = sbm(self.n, self.Psy, directed=True, loops=True, dc=dc, dc_kws=dc_kws)
        for i in range(0, len(self.n)):
            for j in range(0, len(self.n)):
                block = A[
                    (self.vcount[i] - self.n[i]) : self.vcount[i],
                    (self.vcount[j] - self.n[j]) : self.vcount[j],
                ]
                if i == j:
                    block = remove_diagonal(block)
                self.assertTrue(np.isclose(np.mean(block), self.Psy[i, j], atol=0.02))
        self.assertFalse(is_symmetric(A))
        self.assertFalse(is_loopless(A))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(self.n), np.sum(self.n)))
        pass

    def test_sbm_multi_dc_empty_dc_kws(self):
        np.random.seed(self.seed)
        dc = [np.random.rayleigh, np.random.uniform]
        A = sbm(self.n, self.Psy, directed=True, loops=True, dc=dc)
        for i in range(0, len(self.n)):
            for j in range(0, len(self.n)):
                block = A[
                    (self.vcount[i] - self.n[i]) : self.vcount[i],
                    (self.vcount[j] - self.n[j]) : self.vcount[j],
                ]
                if i == j:
                    block = remove_diagonal(block)
                self.assertTrue(np.isclose(np.mean(block), self.Psy[i, j], atol=0.02))
        self.assertFalse(is_symmetric(A))
        self.assertFalse(is_loopless(A))
        # check dimensions
        self.assertTrue(A.shape == (np.sum(self.n), np.sum(self.n)))
        pass

    def test_bad_inputs(self):
        with self.assertRaises(TypeError):
            n = "1"
            sbm(n, self.Psy)

        with self.assertRaises(ValueError):
            n = ["1", 10]
            sbm(n, self.Psy)

        with self.assertRaises(TypeError):
            p = 0.5
            sbm(self.n, p)

        with self.assertRaises(ValueError):
            p = [[0.5]]
            sbm(self.n, p)

        with self.assertRaises(ValueError):
            p = [[5, 5], [4, 4]]
            sbm(self.n, p)

        with self.assertRaises(ValueError):
            p = ["str"]
            sbm(self.n, p)

        with self.assertRaises(TypeError):
            wt = "1"
            sbm(self.n, self.Psy, wt=wt)

        with self.assertRaises(TypeError):
            wt = [[1]]
            sbm(self.n, self.Psy, wt=wt)

        with self.assertRaises(ValueError):
            wtargs = [[1, 1], [1, 1]]
            wt = [[1]]
            sbm(self.n, self.Psy, wt=wt, wtargs=wtargs)

        with self.assertRaises(ValueError):
            wt = [[1, 1], [1, 1]]
            wtargs = [[1, 1]]
            sbm(self.n, self.Psy, wt=wt, wtargs=wtargs)

        with self.assertRaises(TypeError):
            wt = [[1, 1], [1, 1]]
            wtargs = [[1, 1], [1, 1]]
            sbm(self.n, self.Psy, wt=wt, wtargs=wtargs)

        with self.assertRaises(ValueError):
            sbm(self.n, self.Pns)

        with self.assertRaises(ValueError):
            wt = [
                [np.random.uniform, np.random.beta],
                [np.random.uniform, np.random.normal],
            ]
            wtargs = [[1, 1], [1, 1]]
            sbm(self.n, self.Psy, wt=wt, wtargs=wtargs)

        with self.assertRaises(ValueError):
            wt = [
                [np.random.uniform, np.random.uniform],
                [np.random.uniform, np.random.normal],
            ]
            wtargs = [[1, 2], [1, 1]]
            sbm(self.n, self.Psy, wt=wt, wtargs=wtargs)

        with self.assertRaises(TypeError):
            # Check that the paramters are a dict
            dc = np.random.uniform
            dc_kws = [1, 2]
            sbm(self.n, self.Psy, dc=dc, dc_kws=dc_kws)

        with self.assertRaises(ValueError):
            # There are non-numeric elements in p
            dc = ["1"] * sum(self.n)
            sbm(self.n, self.Psy, dc=dc)

        with self.assertRaises(ValueError):
            # dc must have size sum(n)
            dc = [1, 1]
            sbm(self.n, self.Psy, dc=dc)

        with self.assertRaises(ValueError):
            # Values in dc cannot be negative
            dc = -1 * np.ones(sum(self.n))
            sbm(self.n, self.Psy, dc=dc)

        with self.assertWarns(UserWarning):
            # Check that probabilities sum to 1 in each block
            dc = np.ones(sum(self.n))
            sbm(self.n, self.Psy, dc=dc)

        with self.assertRaises(ValueError):
            # dc must be a function, list, or np.array
            dc = {"fail", "me"}
            sbm(self.n, self.Psy, dc=dc)

        with self.assertRaises(ValueError):
            # Check that the paramters are correct len
            dc = [np.random.uniform]
            dc_kws = {}
            sbm(self.n, self.Psy, dc=dc, dc_kws=dc_kws)

        with self.assertRaises(TypeError):
            # dc_kws must be array-like
            dc = [np.random.uniform] * len(self.n)
            dc_kws = {"low": 0, "high": 1}
            sbm(self.n, self.Psy, dc=dc, dc_kws=dc_kws)

        with self.assertRaises(ValueError):
            # dc_kws must be of correct length
            dc = [np.random.uniform] * len(self.n)
            dc_kws = [{}]
            sbm(self.n, self.Psy, dc=dc, dc_kws=dc_kws)

        with self.assertRaises(TypeError):
            # dc_kws must be of correct length
            dc = [np.random.uniform] * len(self.n)
            dc_kws = [1] + [{}] * (len(self.n) - 1)
            sbm(self.n, self.Psy, dc=dc, dc_kws=dc_kws)


class Test_RDPG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = [50, 70]
        cls.Pns = np.vstack(([0.6, 0.2], [0.3, 0.4]))
        # define symmetric probability as evenly weighted
        cls.Psy = np.vstack(([0.6, 0.2], [0.3, 0.4]))
        cls.Psy = symmetrize(cls.Psy)

    def test_dimensions(self):
        X = np.array([[1, 1], [1, 1], [1, 1], [1, 0], [1, 0]])
        A = rdpg(X)
        self.assertTrue(A.shape, (5, 5))

    def test_inputs(self):
        x1 = np.array([[1, 1], [1, 1]])
        x2 = np.array([[1, 1]])
        x3 = np.zeros((2, 2, 2))
        with self.assertRaises(TypeError):
            p_from_latent("hi")  # wrong type
        with self.assertRaises(ValueError):
            p_from_latent(x1, x2)  # dimension mismatch
        with self.assertRaises(ValueError):
            p_from_latent(x3)  # wrong num dimensions
        with self.assertRaises(TypeError):
            sample_edges("XD")  # wrong type
        with self.assertRaises(ValueError):
            sample_edges(x3)  # wrong num dimensions
        with self.assertRaises(ValueError):
            sample_edges(x2)  # wrong shape for P

    def test_er_p_is_close(self):
        np.random.seed(8888)
        X = 0.5 * np.ones((100, 2))
        graphs = []
        P = p_from_latent(X, rescale=True, loops=True)
        for i in range(1000):
            graphs.append(sample_edges(P, directed=True, loops=True))
        graphs = np.stack(graphs)
        self.assertAlmostEqual(np.mean(graphs), 0.5, delta=0.001)
        # mean_graph = np.mean(graphs, axis=0)
        # this only seems to work as n_graphs -> 10000
        # np.testing.assert_allclose(P, mean_graph, atol=0.05)

    def test_mini_sbm_p_is_close(self):
        np.random.seed(8888)
        blocks = np.array([[0.8, 0.1], [0.1, 0.5]])
        X = np.array([[-0.87209812, -0.19860733], [-0.26405006, 0.65595546]])
        graphs = []
        P = p_from_latent(X, rescale=True, loops=True)
        for i in range(10000):
            graphs.append(sample_edges(P, directed=False, loops=True))
        graphs = np.stack(graphs)
        mean_graph = np.mean(graphs, axis=0)
        # this atol should be ~5 stdev away
        np.testing.assert_allclose(blocks, mean_graph, atol=0.025)

    def test_kwarg_passing(self):
        np.random.seed(8888)
        X = 0.5 * np.ones((300, 2))
        g = rdpg(X, rescale=True, loops=True, directed=True)
        self.assertFalse(is_symmetric(g))
        self.assertFalse(is_loopless(g))
        g = rdpg(X, rescale=True, loops=False, directed=True)
        self.assertFalse(is_symmetric(g))
        self.assertTrue(is_loopless(g))
        g = rdpg(X, rescale=True, loops=False, directed=False)
        self.assertTrue(is_symmetric(g))
        self.assertTrue(is_loopless(g))
