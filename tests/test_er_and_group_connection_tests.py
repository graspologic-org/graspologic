import unittest

import numpy as np
from scipy.sparse import csr_matrix

from graspologic.inference.density_test import density_test
from graspologic.inference.group_connection_test import group_connection_test
from graspologic.simulations import er_np, sbm


class TestGroupConnection(unittest.TestCase):
    def test_gctest_works(self):
        np.random.seed(123)
        B1 = np.array([[0.8, 0.6], [0.6, 0.8]])
        B2 = np.array([[0.87, 0.66], [0.66, 0.87]])
        A1, labels1 = sbm([50, 50], B1, return_labels=True)
        A2, labels2 = sbm([60, 60], B2, return_labels=True)

        stat, pvalue, misc = group_connection_test(A1, A2, labels1, labels2)
        self.assertTrue(pvalue <= 0.05)

        stat, pvalue, misc = group_connection_test(
            A1, A2, labels1, labels2, combine_method="fisher", density_adjustment=True
        )
        self.assertTrue(pvalue > 0.05)

    def test_all_kwargs(self):
        B1 = np.array([[0.8, 0.6], [0.6, 0.8]])
        B2 = np.array([[0.87, 0.66], [0.66, 0.87]])
        A1, labels1 = sbm([50, 50], B1, return_labels=True)
        A2, labels2 = sbm([60, 60], B2, return_labels=True)
        stat, pvalue, misc = group_connection_test(
            A1,
            A2,
            labels1,
            labels2,
            combine_method="fisher",
            method="chi2",
            density_adjustment=True,
            correct_method="Bonferroni",
        )
        self.assertTrue(pvalue <= 0.05)
        self.assertTrue(misc["uncorrected_pvalues"].size == 4)
        self.assertTrue(misc["probabilities1"].size == 4)
        self.assertTrue(misc["probabilities2"].size == 4)
        self.assertTrue(misc["observed1"] == A1.count_nonzero())
        self.assertTrue(misc["observed2"] == A2.count_nonzero())
        self.assertTrue(misc["group_counts1"] == [50, 50])
        self.assertTrue(misc["group_counts1"] == [60, 60])
        self.assertTrue(misc["null_ratio"] != 1.0)
        self.assertTrue(misc["n_tests"] == 4)
        self.assertTrue(misc["rejections"] == 4)
        self.assertTrue(misc["corrected_pvalues"].size == 4)

    def test_sparse(self):
        B1 = np.array([[0.8, 0.6], [0.6, 0.8]])
        B2 = np.array([[0.87, 0.66], [0.66, 0.87]])
        A1, labels1 = sbm([50, 50], B1, return_labels=True)
        A2, labels2 = sbm([60, 60], B2, return_labels=True)
        sA1 = csr_matrix(A1)
        sA2 = csr_matrix(A2)

        stat, pvalue, misc = group_connection_test(sA1, sA2, labels1, labels2)
        self.assertTrue(pvalue <= 0.05)

    def test_bad_kwargs(self):
        B1 = np.array([[0.8, 0.6], [0.6, 0.8]])
        B2 = np.array([[0.87, 0.66], [0.66, 0.87]])
        A1, labels1 = sbm([50, 50], B1, return_labels=True)
        A2, labels2 = sbm([60, 60], B2, return_labels=True)
        with self.assertRaises(ValueError):
            group_connection_test(A1, A2, labels1, labels2, method=5)
        with self.assertRaises(ValueError):
            group_connection_test(A1, A2, labels1, labels2, method="hello")
        with self.assertRaises(TypeError):
            group_connection_test("apple", A2, labels1, labels2)
        with self.assertRaises(TypeError):
            group_connection_test(A1, "banana", labels1, labels2)
        with self.assertRaises(TypeError):
            group_connection_test(A1, A2, "orange", labels2)
        with self.assertRaises(TypeError):
            group_connection_test(A1, A2, labels1, "grape")
        with self.assertRaises(ValueError):
            group_connection_test(A1, A2, labels1, [3, 4, 5, 6])
        with self.assertRaises(ValueError):
            group_connection_test(A1, A2, [3, 4, 5, 6], labels2)
        with self.assertRaises(TypeError):
            group_connection_test(A1, A2, labels1, labels2, density_adjustment="apple")
        with self.assertRaises(TypeError):
            group_connection_test(A1, A2, labels1, labels2, combine_method=12)
        with self.assertRaises(TypeError):
            group_connection_test(A1, A2, labels1, labels2, correct_method=42)
        with self.assertRaises(TypeError):
            group_connection_test(A1, A2, labels1, labels2, alpha="apple")


class TestER(unittest.TestCase):
    def test_er(self):
        np.random.seed(234)
        A1 = er_np(500, 0.6)
        A2 = er_np(400, 0.8)
        stat, pvalue, er_misc = density_test(A1, A2)
        self.assertTrue(pvalue <= 0.05)
        A3 = er_np(500, 0.79)
        A4 = er_np(400, 0.8)
        stat, pvalue, er_misc = density_test(A3, A4)
        self.assertTrue(pvalue > 0.05)

    def test_all(self):
        np.random.seed(234)
        A1 = er_np(500, 0.6)
        A2 = er_np(400, 0.8)
        stat, pvalue, er_misc = density_test(A1, A2, method="chi2")
        self.assertTrue(pvalue <= 0.05)
        self.assertTrue(er_misc["probability1"] < 1.0)
        self.assertTrue(er_misc["probability2"] < 1.0)
        self.assertTrue(er_misc["observed1"] == A1.count_nonzero())
        self.assertTrue(er_misc["observed2"] == A2.count_nonzero())

    def test_bad_kwargs(self):
        np.random.seed(234)
        A1 = er_np(500, 0.6)
        A2 = er_np(400, 0.8)
        with self.assertRaises(ValueError):
            density_test(A1, A2, method="hello")
        with self.assertRaises(TypeError):
            density_test(A1, A2, method=5)
        with self.assertRaises(TypeError):
            density_test("hello", A2)
        with self.assertRaises(TypeError):
            density_test(A1, "hello")


if __name__ == "__main__":
    unittest.main()
