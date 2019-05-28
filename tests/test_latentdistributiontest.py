# Bijan Varjavand
# bvarjavand [at] jhu.edu
# 02.26.2019

import unittest

import numpy as np

from graspy.inference import LatentDistributionTest
from graspy.simulations import er_np, sbm


class TestLatentDistributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(123456)
        cls.A1 = er_np(20, 0.3)
        cls.A2 = er_np(20, 0.3)

    def test_fit_p_ase_works(self):
        npt = LatentDistributionTest()
        p = npt.fit(self.A1, self.A2)

    def test_bad_kwargs(self):
        with self.assertRaises(ValueError):
            LatentDistributionTest(n_components=-100)
        with self.assertRaises(ValueError):
            LatentDistributionTest(n_bootstraps=-100)
        with self.assertRaises(TypeError):
            LatentDistributionTest(n_bootstraps=0.5)
        with self.assertRaises(TypeError):
            LatentDistributionTest(n_components=0.5)
        with self.assertRaises(TypeError):
            LatentDistributionTest(bandwidth="oops")

    def test_n_bootstraps(self):
        npt = LatentDistributionTest(n_bootstraps=234, n_components=None)
        npt.fit(self.A1, self.A2)
        self.assertEqual(npt.null_distribution_.shape[0], 234)

    def test_bad_matrix_inputs(self):
        npt = LatentDistributionTest()

        bad_matrix = [[1, 2]]
        with self.assertRaises(TypeError):
            npt.fit(bad_matrix, self.A2)

    def test_directed_inputs(self):
        np.random.seed(2)
        A = er_np(100, 0.3, directed=True)
        B = er_np(100, 0.3, directed=True)

        npt = LatentDistributionTest()
        with self.assertRaises(NotImplementedError):
            npt.fit(A, B)

    def test_different_sizes(self):
        np.random.seed(3)
        A = er_np(50, 0.3)
        B = er_np(100, 0.3)

        npt = LatentDistributionTest()
        with self.assertRaises(ValueError):
            npt.fit(A, B)

    def test_SBM_epsilon(self):
        np.random.seed(12345678)
        B1 = np.array([[0.5, 0.2], [0.2, 0.5]])

        B2 = np.array([[0.7, 0.2], [0.2, 0.7]])
        b_size = 200
        A1 = sbm(2 * [b_size], B1)
        A2 = sbm(2 * [b_size], B1)
        A3 = sbm(2 * [b_size], B2)

        npt_null = LatentDistributionTest(n_components=2, n_bootstraps=100)
        npt_alt = LatentDistributionTest(n_components=2, n_bootstraps=100)
        p_null = npt_null.fit(A1, A2)
        p_alt = npt_alt.fit(A1, A3)
        self.assertTrue(p_null > 0.05)
        self.assertTrue(p_alt <= 0.05)


if __name__ == "__main__":
    unittest.main()
