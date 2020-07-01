import unittest
from graspy.simulations import siem
from graspy.models import SIEMEstimator
import numpy as np
from copy import deepcopy


def modular_edges(n):
    """
    A function for generating modular sbm edge communities.
    """
    m = int(n / 2)
    edge_comm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if ((i < m) & (j < m)) or ((i >= m) & (j >= m)):
                edge_comm[i, j] = 1
            else:
                edge_comm[i, j] = 2
    return edge_comm


def nuis_edges(n):
    """
    A function for generating doubly modular sbm.
    """
    m = int(n / 2)
    m4 = int(7 * n / 8)
    m3 = int(5 * n / 8)
    m2 = int(3 * n / 8)
    m1 = int(1 * n / 8)
    edge_comm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if ((i < m) & (j < m)) or ((i >= m) & (j >= m)):
                edge_comm[i, j] = 1
            elif (((i >= m3) & (i <= m4)) & ((j >= m1) & (j <= m2))) or (
                ((i >= m1) & (i <= m2)) & ((j >= m3) & (j <= m4))
            ):
                edge_comm[i, j] = 3
            else:
                edge_comm[i, j] = 2
    return edge_comm


def diag_edges(n):
    """
    A function for generating diagonal SIEM edge communities.
    """
    m = int(n / 2)
    edge_comm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i == j + m) or (j == i + m):
                edge_comm[i, j] = 1
            else:
                edge_comm[i, j] = 2
    return edge_comm


class Test_Model_Fit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 100
        # set up edge-communities for relevant graphs
        cls.modular_edges = modular_edges(cls.n)
        cls.model = SIEMEstimator()

    def test_uw_case(self):
        p = [0.75, 0.5]
        np.random.seed(1)
        graph = siem(self.n, p, self.modular_edges, loops=True)
        model = deepcopy(self.model)
        model.fit(graph, self.modular_edges)
        # check that weights saved are appropriate
        self.assertTrue(set(model.model.keys()) == set([1.0, 2.0]))
        self.assertTrue(
            np.allclose((model.model[1.0]["weights"]).mean(), p[0], atol=0.02)
        )
        self.assertTrue(
            np.allclose((model.model[2.0]["weights"]).mean(), p[1], atol=0.02)
        )
        # check that the edges indexed for the particular communities are appropriate
        self.assertTrue(model.model[1.0]["edges"]) == np.where(
            graph[self.modular_edges == 1]
        )
        self.assertTrue(model.model[2.0]["edges"]) == np.where(
            graph[self.modular_edges == 2]
        )
        pass

    def test_bad_inputs(self):
        # graph is not a nd array or numpy object
        with self.assertRaises(TypeError):
            model = deepcopy(self.model)
            model.fit("test", self.modular_edges)

        # graph is not square
        graph = np.ones((self.n, self.n + 1))
        graph[2, 2] = 0
        with self.assertRaises(ValueError):
            model = deepcopy(self.model)
            model.fit(graph, self.modular_edges)

        np.random.seed(1)
        graph = siem(self.n, 0.5, self.modular_edges, loops=True)
        graph[2, 2] = np.inf
        # graph contains non-finite entries
        with self.assertRaises(ValueError):
            model = deepcopy(self.model)
            model.fit(graph, self.modular_edges)

        np.random.seed(1)
        graph = siem(self.n, 0.5, self.modular_edges, loops=True)
        # edge_comm is not square
        with self.assertRaises(ValueError):
            model = deepcopy(self.model)
            model.fit(graph, np.ones((self.n, self.n + 1)))

        np.random.seed(1)
        graph = siem(self.n + 2, 0.5, modular_edges(self.n + 2), loops=True)
        # graph and edge_comm are both square, but not same size
        with self.assertRaises(ValueError):
            model = deepcopy(self.model)
            model.fit(graph, self.modular_edges)

        graph = siem(self.n, 0.5, self.modular_edges, loops=True)
        model = deepcopy(self.model)
        model.fit(graph, self.modular_edges)
        # fitting twice causes warning
        with self.assertWarns(Warning):
            model_double = deepcopy(model)
            model_double.fit(graph, self.modular_edges)
        pass


class Test_Model_Summary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 100
        # set up edge-communities for relevant graphs
        cls.modular_edges = modular_edges(cls.n)
        cls.model = SIEMEstimator()

    def test_summary_func(self):
        np.random.seed(1)
        graph = siem(
            self.n,
            1,
            self.modular_edges,
            loops=True,
            wt=[np.random.normal, np.random.normal],
            wtargs=[{"loc": 0, "scale": 1}, {"loc": 2, "scale": 1}],
        )
        model = deepcopy(self.model)
        model.fit(graph, self.modular_edges)
        msum = model.summarize(
            {"loc": np.mean, "scale": np.std}, {"loc": {}, "scale": {}},
        )
        self.assertTrue(np.allclose(msum[1.0]["loc"], 0, atol=0.02))
        self.assertTrue(np.allclose(msum[1.0]["scale"], 1, atol=0.05))
        self.assertTrue(np.allclose(msum[2.0]["loc"], 2, atol=0.02))
        self.assertTrue(np.allclose(msum[2.0]["scale"], 1, atol=0.05))
        pass

    def test_bad_inputs(self):
        np.random.seed(1)
        graph = siem(self.n, 0.5, self.modular_edges, loops=True)
        model = deepcopy(self.model)
        # summarize before fitting model raises error
        with self.assertRaises(UnboundLocalError):
            model.summarize({}, {})

        model.fit(graph, self.modular_edges)

        # summarize with wt and wtargs not having same key names
        with self.assertRaises(ValueError):
            model.summarize({"mean": np.mean}, {"meaner": {"a": None}})

        # summarize with wt not being a dictionary of callables
        with self.assertRaises(TypeError):
            model.summarize({"mean": 4}, {"mean": {}})

        # summarize with wtargs not being a dictionary of sub-dictionaries
        with self.assertRaises(TypeError):
            model.summarize({"mean": np.mean}, {"mean": "a"})
        pass


class Test_Model_Comparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 100
        # set up edge-communities for relevant graphs
        cls.modular_edges = modular_edges(cls.n)
        cls.model = SIEMEstimator()

    def test_cmp(self):
        np.random.seed(1)
        graph = siem(
            self.n,
            1,
            self.modular_edges,
            loops=True,
            wt=[np.random.normal, np.random.normal],
            wtargs=[{"loc": 0, "scale": 1}, {"loc": 2, "scale": 1}],
        )
        model = deepcopy(self.model)
        model.fit(graph, self.modular_edges)
        test = model.compare(1.0, 2.0, methodargs={"alternative": "two-sided"})
        self.assertTrue(np.allclose(test[1], 0, atol=0.02))
        test = model.compare(1.0, 2.0, methodargs={"alternative": "less"})
        self.assertTrue(np.allclose(test[1], 0, atol=0.01))
        test = model.compare(1.0, 2.0, methodargs={"alternative": "greater"})
        self.assertTrue(np.allclose(test[1], 1, atol=0.01))
        # communities are passed in in appropriate ordering
        test1 = model.compare(1.0, 2.0, methodargs={"alternative": "greater"})
        test2 = model.compare(2.0, 1.0, methodargs={"alternative": "less"})
        self.assertTrue(test1[1] == test2[1])
        pass

    def test_bad_inputs(self):
        np.random.seed(1)
        graph = siem(self.n, 0.5, self.modular_edges, loops=True)
        model = deepcopy(self.model)
        # compare before fitting model raises error
        with self.assertRaises(UnboundLocalError):
            model.compare("test1", "test2")
        model.fit(graph, self.modular_edges)
        # pass communities that are not community names
        with self.assertRaises(ValueError):
            model.compare("1", 2.0)
        with self.assertRaises(ValueError):
            model.compare(1.0, "2")
        # wt is not callable
        with self.assertRaises(TypeError):
            model.compare(1.0, 2.0, "mww")
        # wtargs is not a dictionary
        with self.assertRaises(TypeError):
            model.compare(1.0, 2.0, methodargs="two-sided")
        pass
