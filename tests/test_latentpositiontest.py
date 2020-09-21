# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
import numpy as np
from graspy.inference import LatentPositionTest
from graspy.simulations import er_np, sbm
from graspy.utils import *


class TestLatentPositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1234556)
        cls.A1 = er_np(20, 0.3)
        cls.A2 = er_np(20, 0.3)

    def test_fit_ase_works(self):
        spt = LatentPositionTest()
        assert spt.fit(self.A1, self.A2) is spt

    def test_fit_omni_works(self):
        spt = LatentPositionTest(embedding="omnibus")
        assert spt.fit(self.A1, self.A2) is spt

    def test_fit_predict_ase_works(self):
        spt = LatentPositionTest()
        p = spt.fit_predict(self.A1, self.A2)
        assert float(p) <= 1 and float(p) >= 0

    def test_bad_kwargs(self):
        with self.assertRaises(ValueError):
            LatentPositionTest(n_components=-100)
        with self.assertRaises(ValueError):
            LatentPositionTest(n_components=-100)
        with self.assertRaises(ValueError):
            LatentPositionTest(test_case="oops")
        with self.assertRaises(ValueError):
            LatentPositionTest(n_bootstraps=-100)
        with self.assertRaises(ValueError):
            LatentPositionTest(embedding="oops")
        with self.assertRaises(TypeError):
            LatentPositionTest(n_bootstraps=0.5)
        with self.assertRaises(TypeError):
            LatentPositionTest(n_components=0.5)
        with self.assertRaises(TypeError):
            LatentPositionTest(embedding=6)
        with self.assertRaises(TypeError):
            LatentPositionTest(test_case=6)

    def test_n_bootstraps(self):
        spt = LatentPositionTest(n_bootstraps=234, n_components=None)
        spt.fit(self.A1, self.A2)
        self.assertEqual(spt.null_distribution_1_.shape[0], 234)

    def test_bad_matrix_inputs(self):
        spt = LatentPositionTest()
        A1 = self.A1.copy()
        A1[2, 0] = 1  # make asymmetric
        with self.assertRaises(NotImplementedError):  # TODO : remove when we implement
            spt.fit(A1, self.A2)

        bad_matrix = [[1, 2]]
        with self.assertRaises(TypeError):
            spt.fit(bad_matrix, self.A2)

        with self.assertRaises(ValueError):
            spt.fit(self.A1[:2, :2], self.A2)

    def test_rotation_norm(self):
        # two triangles rotated by 90 degrees
        points1 = np.array([[0, 0], [3, 0], [3, -2]])
        rotation = np.array([[0, 1], [-1, 0]])
        points2 = np.dot(points1, rotation)

        spt = LatentPositionTest(embedding="ase", test_case="rotation")
        n = spt._difference_norm(points1, points2)
        self.assertAlmostEqual(n, 0)

    def test_diagonal_rotation_norm(self):
        # triangle in 2d
        points1 = np.array([[0, 0], [3, 0], [3, -2]], dtype=np.float64)
        rotation = np.array([[0, 1], [-1, 0]])
        # rotated 90 degrees
        points2 = np.dot(points1, rotation)
        # diagonally scaled
        diagonal = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 2]])
        points2 = np.dot(diagonal, points2)

        spt = LatentPositionTest(embedding="ase", test_case="diagonal-rotation")
        n = spt._difference_norm(points1, points2)
        self.assertAlmostEqual(n, 0)

    def test_scalar_rotation_norm(self):
        # triangle in 2d
        points1 = np.array([[0, 0], [3, 0], [3, -2]], dtype=np.float64)
        rotation = np.array([[0, 1], [-1, 0]])
        # rotated 90 degrees
        points2 = np.dot(points1, rotation)
        # scaled
        points2 = 2 * points2

        spt = LatentPositionTest(embedding="ase", test_case="scalar-rotation")
        n = spt._difference_norm(points1, points2)
        self.assertAlmostEqual(n, 0)

    def test_SBM_epsilon(self):
        np.random.seed(12345678)
        B1 = np.array([[0.5, 0.2], [0.2, 0.5]])

        B2 = np.array([[0.7, 0.2], [0.2, 0.7]])
        b_size = 200
        A1 = sbm(2 * [b_size], B1)
        A2 = sbm(2 * [b_size], B1)
        A3 = sbm(2 * [b_size], B2)

        spt_null = LatentPositionTest(n_components=2, n_bootstraps=100)
        spt_alt = LatentPositionTest(n_components=2, n_bootstraps=100)
        p_null = spt_null.fit_predict(A1, A2)
        p_alt = spt_alt.fit_predict(A1, A3)
        self.assertTrue(p_null > 0.05)
        self.assertTrue(p_alt <= 0.05)


if __name__ == "__main__":
    unittest.main()
