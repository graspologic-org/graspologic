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
        cls.r = 0.3
        cls.P = cls.p * np.ones((cls.n, cls.n))

    def test_bad_input(self):
        with self.assertRaises(TypeError):
            P = "0.5"
            sample_edges_corr(P, self.r, directed=False, loops=False)

        with self.assertRaises(TypeError):
            r = 10
            sample_edges_corr(self.P, r, directed=False, loops=False)

        with self.assertRaises(TypeError):
            r = "1"
            sample_edges_corr(self.P, r, directed=False, loops=False)

        with self.assertRaises(ValueError):
            r = -0.5
            sample_edges_corr(self.P, r, directed=False, loops=False)

        with self.assertRaises(ValueError):
            r = 5.0
            sample_edges_corr(self.P, r, directed=False, loops=False)

        with pytest.raises(TypeError):
            sample_edges_corr(self.P, self.r, directed="hey", loops=False)

        with pytest.raises(TypeError):
            sample_edges_corr(self.P, self.r, directed=False, loops=6)

    def test_sample_edges_corr(self):
        # P = self.p * np.ones((self.n, self.n))
        P = self.P
        g1, g2 = sample_edges_corr(P, self.r, directed=False, loops=False)

        # check the 1 probability of the output binary matrix
        # should be near to the input P matrix
        self.assertTrue(
            np.isclose(self.p, g2.sum() / (self.n * (self.n - 1)), atol=0.05)
        )

        # check rho in graph2
        add = g1 + g2
        add[add != 2] = 0
        output_prob = add.sum() / (2 * self.n * (self.n - 1))
        output_r = np.abs(output_prob - self.p ** 2) / (self.p - self.p ** 2)
        self.assertTrue(np.isclose(self.r, output_r, atol=0.06))

        # check the similarity of g1 and g2
        judge = g1 == g2
        judge.astype(int)
        output_sim = (np.sum(judge) - self.n) / (self.n * (self.n - 1))
        expected_sim = self.p * (self.p + self.r * (1 - self.p)) + (1 - self.p) * (
            1 - self.p * (1 - self.r)
        )
        self.assertTrue(np.isclose(expected_sim, output_sim, atol=0.05))

        # check the dimension of input P and Rho
        self.assertTrue(g1.shape == (self.n, self.n))
        self.assertTrue(g2.shape == (self.n, self.n))

    def test_sample_edges_er_corr(self):
        g1, g2 = sample_edges_er_corr(
            self.n, self.p, self.r, directed=False, loops=False
        )
        # check the 1 probability of the output binary matrix
        # should be near to the input P matrix
        self.assertTrue(
            np.isclose(self.p, g2.sum() / (self.n * (self.n - 1)), atol=0.05)
        )

        # check rho in graph2
        add = g1 + g2
        add[add != 2] = 0
        output_prob = add.sum() / (2 * self.n * (self.n - 1))
        output_r = np.abs(output_prob - self.p ** 2) / (self.p - self.p ** 2)
        self.assertTrue(np.isclose(self.r, output_r, atol=0.06))

        # check the similarity of g1 and g2
        judge = g1 == g2
        judge.astype(int)
        output_sim = (np.sum(judge) - self.n) / (self.n * (self.n - 1))
        expected_sim = self.p * (self.p + self.r * (1 - self.p)) + (1 - self.p) * (
            1 - self.p * (1 - self.r)
        )
        self.assertTrue(np.isclose(expected_sim, output_sim, atol=0.05))

        # check the dimension of input P and Rho
        self.assertTrue(g1.shape == (self.n, self.n))
        self.assertTrue(g2.shape == (self.n, self.n))
