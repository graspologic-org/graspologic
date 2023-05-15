import unittest

import numpy as np
from scipy.sparse import csr_array

from graspologic.inference import density_test, group_connection_test
from graspologic.simulations import er_np, sbm


class TestGroupConnection(unittest.TestCase):
    def test_gctest_works(self):
        np.random.seed(8888)
        B1 = np.array([[0.8, 0.6], [0.6, 0.8]])
        B2 = 0.8 * B1
        A1, labels1 = sbm([50, 50], B1, return_labels=True)
        A2, labels2 = sbm([60, 60], B2, return_labels=True)
        stat, pvalue, misc = group_connection_test(
            A1, A2, labels1, labels2, density_adjustment=True
        )
        self.assertTrue(pvalue > 0.05)

    def test_all_kwargs(self):
        B1 = np.array([[0.4, 0.6], [0.6, 0.8]])
        B2 = np.array([[0.9, 0.4], [0.2, 0.865]])
        A1, labels1 = sbm([60, 60], B1, return_labels=True, directed=True)
        A2, labels2 = sbm([50, 50], B2, return_labels=True, directed=True)
        stat, pvalue, misc = group_connection_test(
            A1,
            A2,
            labels1,
            labels2,
            combine_method="tippett",
            method="score",
            correct_method="Bonferroni",
            density_adjustment=True,
        )
        self.assertTrue(pvalue < 0.05)
        self.assertTrue(misc["uncorrected_pvalues"].size == 4)
        self.assertTrue(misc["probabilities1"].size == 4)
        self.assertTrue(misc["probabilities2"].size == 4)
        self.assertTrue(np.sum(misc["observed1"].to_numpy()) == np.count_nonzero(A1))
        self.assertTrue(np.sum(misc["observed2"].to_numpy()) == np.count_nonzero(A2))
        self.assertTrue(misc["null_ratio"] != 1.0)
        self.assertTrue(misc["n_tests"] == 4)
        self.assertTrue(misc["rejections"].to_numpy().size == 4)
        self.assertTrue(misc["corrected_pvalues"].size == 4)

    def test_sparse(self):
        B1 = np.array([[0.8, 0.6], [0.6, 0.8]])
        B2 = np.array([[0.87, 0.66], [0.66, 0.87]])
        A1, labels1 = sbm([50, 50], B1, return_labels=True)
        A2, labels2 = sbm([60, 60], B2, return_labels=True)
        sA1 = csr_array(A1)
        sA2 = csr_array(A2)

        stat, pvalue, misc = group_connection_test(sA1, sA2, labels1, labels2)
        self.assertTrue(pvalue <= 0.05)


class TestER(unittest.TestCase):
    def test_er(self):
        np.random.seed(234)
        A1 = er_np(500, 0.6)
        A2 = er_np(400, 0.8)
        stat, pvalue, er_misc = density_test(A1, A2)
        self.assertTrue(pvalue <= 0.05)
        A3 = er_np(500, 0.8)
        A4 = er_np(400, 0.8)
        stat, pvalue, er_misc = density_test(A3, A4)
        self.assertTrue(pvalue > 0.05)

    def test_all(self):
        np.random.seed(234)
        A1 = er_np(500, 0.6)
        A2 = er_np(400, 0.8)
        stat, pvalue, er_misc = density_test(A1, A2, method="chi2")
        self.assertTrue(pvalue <= 0.05)
        self.assertTrue(er_misc["probability1"].to_numpy() < 1.0)
        self.assertTrue(er_misc["probability2"].to_numpy() < 1.0)
        self.assertTrue(er_misc["observed1"].to_numpy() == np.count_nonzero(A1))
        self.assertTrue(er_misc["observed2"].to_numpy() == np.count_nonzero(A2))


if __name__ == "__main__":
    unittest.main()
