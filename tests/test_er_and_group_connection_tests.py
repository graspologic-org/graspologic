import unittest

import numpy as np
from scipy.sparse import csr_matrix

from graspologic.inference.density_test import density_test
from graspologic.inference.group_connection_test import group_connection_test
from graspologic.simulations import er_np, sbm


class TestGroupConnection(unittest.TestCase):
    def test_gctest_works(self):
        np.random.seed(123)
        B1 = np.array([[0.5, 0.2], [0.2, 0.5]])
        B2 = np.array([[0.7, 0.2], [0.2, 0.7]])
        A1, labels1 = sbm([250, 250], B1, return_labels=True)
        A2, labels2 = sbm([250, 250], B2, return_labels=True)
        with self.assertRaises(ValueError):
            group_connection_test(A1, A2, labels1, labels2, method=5)

        with self.assertRaises(ValueError):
            group_connection_test(A1, A2, labels1, labels2, method="hello")

        stat, pvalue, misc = group_connection_test(A1, A2, labels1, labels2)
        self.assertTrue(pvalue > 0.05)
        self.assertTrue(pvalue <= 0.05)


class TestER(unittest.TestCase):
    def test_er(self):
        np.random.seed(234)
        A1 = er_np(500, 0.6)
        A2 = er_np(400, 0.55)
        with self.assertRaises(ValueError):
            density_test(A1, A2, method="hello")
        stat, pvalue, er_misc = density_test(A1, A2)
        self.assertTrue(pvalue > 0.05)
        self.assertTrue(pvalue <= 0.05)


if __name__ == "__main__":
    unittest.main()
