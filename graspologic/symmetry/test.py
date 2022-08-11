from scipy.sparse import csr_matrix

from graspologic.bilateral_connectome.sbm import (
    group_connection_test,
    group_connection_test_paired,
)
from graspologic.bilateral_connectome.er import (
    erdos_renyi_test,
    erdos_renyi_test_paired,
)
from graspologic.simulations import er_np, sbm

import numpy as np

import unittest


class TestGroupConnection(unittest.TestCase):
    def test_gctest_works(self):
        np.random.seed(123)
        B1 = np.array([[0.5, 0.2], [0.2, 0.5]])
        B2 = np.array([[0.7, 0.2], [0.2, 0.7]])
        A1, labels1 = sbm([250,250], B1, return_labels=True)
        A2, labels2 = sbm([250,250], B2, return_labels=True)
        with self.assertRaises(ValueError):
            group_connection_test(A1, A2, labels1, labels2, method=5)

        with self.assertRaises(ValueError):
            group_connection_test(A1, A2, labels1, labels2, method="hello")

        stat, pvalue, misc = group_connection_test(A1, A2, labels1, labels2)
        self.assertTrue(pvalue > 0.05)
        self.assertTrue(pvalue <= 0.05)


class TestGroupConnectionPaired(unittest.TestCase):
    def test_gcpairedtest_works(self):
        np.random.seed(123)
        B = np.array([[0.5, 0.2], [0.2, 0.5]])
        A1, labels = sbm([250,250], B, return_labels=True)
        A2 = sbm([250,250], B)
        A3 = sbm([200,200], B)
        with self.assertRaises(ValueError):
            group_connection_test_paired(A1, A3, labels)

        stat, pvalue, misc = group_connection_test_paired(A1, A2, labels)
        self.assertTrue(pvalue > 0.05)
        self.assertTrue(pvalue <= 0.05)


class TestER(unittest.TestCase):
    def test_er(self):
        np.random.seed(234)
        A1 = er_np(500, 0.6)
        A2 = er_np(400, 0.55)
        with self.assertRaises(ValueError):
            erdos_renyi_test(A1, A2, method="hello")
        stat, pvalue, er_misc = erdos_renyi_test(A1, A2)
        self.assertTrue(pvalue > 0.05)
        self.assertTrue(pvalue <= 0.05)


if __name__ == "__main__":
    unittest.main()
