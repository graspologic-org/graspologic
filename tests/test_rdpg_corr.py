import unittest
from graspy.simulations.simulations import sample_edges, rdpg, p_from_latent
from graspy.simulations.rdpg_corr import rdpg_corr
import numpy as np
import pytest
import warnings


class Test_RDPG_Corr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = 0.3
        cls.Y = None
        cls.X = np.random.dirichlet([20, 20], size=300)

    def test_dimensions(self):
        A, B = rdpg_corr(
            self.X, self.Y, self.r, rescale=False, directed=False, loops=False
        )
        self.assertTrue(A.shape, (300, 300))
        self.assertTrue(B.shape, (300, 300))

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

    def test_rdpg_corr(self):
        g1, g2 = rdpg_corr(
            self.X, self.Y, self.r, rescale=False, directed=False, loops=False
        )

        # check rho
        g1 = g1[np.where(~np.eye(g1.shape[0], dtype=bool))]
        g2 = g2[np.where(~np.eye(g2.shape[0], dtype=bool))]
        correlation = np.corrcoef(g1, g2)[0, 1]
        self.assertTrue(np.isclose(correlation, self.r, atol=0.05))
