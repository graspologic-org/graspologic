# Bijan Varjavand
# bvarjavand [at] jhu.edu
# 02.26.2019

import unittest

import numpy as np

from graspy.inference import LatentDistributionTest
from graspy.inference.dists import euclidean, gaussian
from graspy.simulations import er_np, sbm


class TestLatentDistributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(123456)
        cls.tests = ["Dcorr", "MGC"]
        cls.dists = [euclidean, gaussian]
        cls.A1 = er_np(20, 0.3)
        cls.A2 = er_np(20, 0.3)

    def test_fit_p_ase_works(self):
        for dist in self.dists:
            for test in self.tests:
                ldt = LatentDistributionTest(test, dist, n_bootstraps=10)
                p = ldt.fit(self.A1, self.A2)

    def test_bad_kwargs(self):
        with self.assertRaises(ValueError):
            LatentDistributionTest(test="foo")
        with self.assertRaises(ValueError):
            LatentDistributionTest(test="Dcorr", n_components=-100)
        with self.assertRaises(ValueError):
            LatentDistributionTest(test="Dcorr", n_bootstraps=-100)
        with self.assertRaises(ValueError):
            LatentDistributionTest(test="Dcorr", num_workers=-1)
        with self.assertRaises(TypeError):
            LatentDistributionTest(test=0)
        with self.assertRaises(TypeError):
            LatentDistributionTest(test="Dcorr", distance=0)
        with self.assertRaises(TypeError):
            LatentDistributionTest(test="Dcorr", n_bootstraps=0.5)
        with self.assertRaises(TypeError):
            LatentDistributionTest(test="Dcorr", n_components=0.5)
        with self.assertRaises(TypeError):
            LatentDistributionTest(test="Dcorr", num_workers=0.5)
        with self.assertRaises(NotImplementedError):
            LatentDistributionTest(test="Dcorr", num_workers=4)

    def test_n_bootstraps(self):
        for test in self.tests:
            ldt = LatentDistributionTest(test, n_bootstraps=123)
            ldt.fit(self.A1, self.A2, return_null_dist=True)
            self.assertEqual(ldt.null_distribution_.shape[0], 123)

    def test_bad_matrix_inputs(self):
        ldt = LatentDistributionTest("Dcorr")

        bad_matrix = [[1, 2]]
        with self.assertRaises(TypeError):
            ldt.fit(bad_matrix, self.A2)

    def test_directed_inputs(self):
        np.random.seed(2)
        A = er_np(100, 0.3, directed=True)
        B = er_np(100, 0.3, directed=True)

        ldt = LatentDistributionTest("Dcorr")
        with self.assertRaises(NotImplementedError):
            p = ldt.fit(A, B)
        # self.assertTrue(p > 0.05)

    def test_SBM_euclidean(self):
        np.random.seed(12345678)
        B1 = np.array([[0.5, 0.2], [0.2, 0.5]])

        B2 = np.array([[0.7, 0.2], [0.2, 0.7]])
        b_size = 200
        A1 = sbm(2 * [b_size], B1)
        A2 = sbm(2 * [b_size], B1)
        A3 = sbm(2 * [b_size], B2)

        for test in self.tests:
            ldt_null = LatentDistributionTest(
                test, euclidean, n_components=2, n_bootstraps=50
            )
            ldt_alt = LatentDistributionTest(
                test, euclidean, n_components=2, n_bootstraps=50
            )
            p_null = ldt_null.fit(A1, A2)
            p_alt = ldt_alt.fit(A1, A3)
            self.assertTrue(p_null > 0.05)
            self.assertTrue(p_alt <= 0.05)

    def test_SBM_gaussian(self):
        np.random.seed(12345678)
        B1 = np.array([[0.5, 0.2], [0.2, 0.5]])

        B2 = np.array([[0.7, 0.2], [0.2, 0.7]])
        b_size = 200
        A1 = sbm(2 * [b_size], B1)
        A2 = sbm(2 * [b_size], B1)
        A3 = sbm(2 * [b_size], B2)

        for test in self.tests:
            ldt_null = LatentDistributionTest(
                test, gaussian, n_components=2, n_bootstraps=50
            )
            ldt_alt = LatentDistributionTest(
                test, gaussian, n_components=2, n_bootstraps=50
            )
            p_null = ldt_null.fit(A1, A2)
            p_alt = ldt_alt.fit(A1, A3)
            self.assertTrue(p_null > 0.05)
            self.assertTrue(p_alt <= 0.05)


if __name__ == "__main__":
    unittest.main()
