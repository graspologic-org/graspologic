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
        npt = LatentDistributionTest(method='dcorr')
        p = npt.fit(self.A1, self.A2)

    def test_fit_mgc_works(self):
        npt = LatentDistributionTest(method='mgc')
        p = npt.fit(self.A1, self.A2)

    def test_bad_kwargs(self):
        with self.assertRaises(ValueError):
            LatentDistributionTest(n_components=-100)
        with self.assertRaises(TypeError):
            LatentDistributionTest(n_components=0.5)
        with self.assertRaises(ValueError):
            LatentDistributionTest(method="oops")

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
        p = npt.fit(A, B)
        self.assertTrue(p > 0.05)

    def test_different_sizes(self):
        np.random.seed(3)
        A = er_np(50, 0.3)
        B = er_np(100, 0.3)
        npt = LatentDistributionTest(method='dcorr')
        p = npt.fit(A,B)
        #self.assertTrue(p < 0.05)
        # CURRENTLY THIS IS SKETCHY SINCE WE KNOW DCORR/MGC IS AN INVALID TEST FOR N != M
        # FOR NOW WE SHOULD LET PEOPLE DO THIS, BUT RESULTS ARE NOT TO BE TRUSTED FOR SMALL N OR N NOT ~ M

    def test_SBM_epsilon(self):
        np.random.seed(12345678)
        B1 = np.array([[0.5, 0.2], [0.2, 0.5]])

        B2 = np.array([[0.7, 0.2], [0.2, 0.7]])
        b_size = 200
        A1 = sbm(2 * [b_size], B1)
        A2 = sbm(2 * [b_size], B1)
        A3 = sbm(2 * [b_size], B2)

        npt_null = LatentDistributionTest(n_components=2, method='dcorr')
        npt_alt = LatentDistributionTest(n_components=2, method='dcorr')
        p_null = npt_null.fit(A1, A2)
        p_alt = npt_alt.fit(A1, A3)
        print('null:', p_null, 'alt:', p_alt)
        self.assertTrue(p_null > 0.05)
        self.assertTrue(p_alt <= 0.05)


if __name__ == "__main__":
    unittest.main()
