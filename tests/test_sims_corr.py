import unittest
from graspy.simulations.simulations_corr import sample_edges_corr, sample_edges_er_corr
import numpy as np
import pytest
import warnings


class Test_Sample_Corr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 500
        cls.p = 0.5
        cls.rho = 0.3

    def test_sample_edges_corr(self):
        P = self.p * np.ones((self.n, self.n))
        R = self.rho * np.ones((self.n, self.n))
        g1, g2 = sample_edges_corr(P, R, directed=False, loops=False)
        # check the 1 probability of the output binary matrix
        # should be near to the input P matrix
        self.assertTrue(
            np.isclose(self.p, g2.sum() / (self.n * (self.n - 1)), atol=0.05)
        )

        # check rho in graph2
        add = g1 + g2
        add[add != 2] = 0
        real_prob = add.sum() / (2 * self.n * (self.n - 1))
        real_rho = np.abs(real_prob - self.p ** 2) / (self.p - self.p ** 2)
        self.assertTrue(np.isclose(self.rho, real_rho, atol=0.06))

        # check the similarity of g1 and g2
        judge = g1 == g2
        judge.astype(int)
        real_sim = (np.sum(judge) - self.n) / (self.n * (self.n - 1))
        exp_sim = self.p * (self.p + self.rho * (1 - self.p)) + (1 - self.p) * (
            1 - self.p * (1 - self.rho)
        )
        self.assertTrue(np.isclose(exp_sim, real_sim, atol=0.05))

        # check the dimension of input P and Rho
        self.assertTrue(g1.shape == (self.n, self.n))
        self.assertTrue(g2.shape == (self.n, self.n))

    def test_sample_edges_er_corr(self):
        P = self.p * np.ones((self.n, self.n))
        R = self.rho * np.ones((self.n, self.n))
        g1, g2 = sample_edges_er_corr(self.n, self.p, R, directed=False, loops=False)
        # check the 1 probability of the output binary matrix
        # should be near to the input P matrix
        self.assertTrue(
            np.isclose(self.p, g2.sum() / (self.n * (self.n - 1)), atol=0.05)
        )

        # check rho in graph2
        add = g1 + g2
        add[add != 2] = 0
        real_prob = add.sum() / (2 * self.n * (self.n - 1))
        real_rho = np.abs(real_prob - self.p ** 2) / (self.p - self.p ** 2)
        self.assertTrue(np.isclose(self.rho, real_rho, atol=0.06))

        # check the similarity of g1 and g2
        judge = g1 == g2
        judge.astype(int)
        real_sim = (np.sum(judge) - self.n) / (self.n * (self.n - 1))
        exp_sim = self.p * (self.p + self.rho * (1 - self.p)) + (1 - self.p) * (
            1 - self.p * (1 - self.rho)
        )
        self.assertTrue(np.isclose(exp_sim, real_sim, atol=0.05))

        # check the dimension of input P and Rho
        self.assertTrue(g1.shape == (self.n, self.n))
        self.assertTrue(g2.shape == (self.n, self.n))
