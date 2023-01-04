# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np

from graspologic.inference import latent_position_test
from graspologic.inference.latent_position_test import _difference_norm
from graspologic.simulations import er_np, sbm


class TestLatentPositionTest(unittest.TestCase):
    @classmethod
    def test_ase_works(self):
        np.random.seed(1234556)
        A1 = er_np(5, 0.8)
        A2 = er_np(5, 0.8)
        lpt = latent_position_test(A1, A2)

    def test_omni_works(self):
        np.random.seed(1234556)
        A1 = er_np(5, 0.8)
        A2 = er_np(5, 0.8)
        lpt = latent_position_test(A1, A2, embedding="omnibus")

    def test_bad_kwargs(self):
        np.random.seed(1234556)
        A1 = er_np(5, 0.8)
        A2 = er_np(5, 0.8)

        with self.assertRaises(ValueError):
            latent_position_test(A1, A2, n_components=-100)
        with self.assertRaises(ValueError):
            latent_position_test(A1, A2, test_case="oops")
        with self.assertRaises(ValueError):
            latent_position_test(A1, A2, n_bootstraps=-100)
        with self.assertRaises(ValueError):
            latent_position_test(A1, A2, embedding="oops")
        with self.assertRaises(TypeError):
            latent_position_test(A1, A2, n_bootstraps=0.5)
        with self.assertRaises(TypeError):
            latent_position_test(A1, A2, n_components=0.5)
        with self.assertRaises(TypeError):
            latent_position_test(A1, A2, embedding=6)
        with self.assertRaises(TypeError):
            latent_position_test(A1, A2, test_case=6)
        with self.assertRaises(TypeError):
            latent_position_test(A1, A2, workers="oops")

    def test_n_bootstraps(self):
        np.random.seed(1234556)
        A1 = er_np(5, 0.8)
        A2 = er_np(5, 0.8)

        lpt = latent_position_test(A1, A2, n_bootstraps=234, n_components=None)
        assert lpt[2]["null_distribution_1"].shape[0] == 234

    def test_bad_matrix_inputs(self):
        np.random.seed(1234556)
        A1 = er_np(5, 0.8)
        A2 = er_np(5, 0.8)
        A1[2, 0] = 1  # make asymmetric
        A1[0, 2] = 0
        with self.assertRaises(NotImplementedError):  # TODO : remove when we implement
            latent_position_test(A1, A2)

        bad_matrix = [[1, 2]]
        with self.assertRaises(TypeError):
            latent_position_test(bad_matrix, A2)

        with self.assertRaises(ValueError):
            latent_position_test(A1[:2, :2], A2)

    def test_rotation_norm(self):
        # two triangles rotated by 90 degrees
        points1 = np.array([[0, 0], [3, 0], [3, -2]])
        rotation = np.array([[0, 1], [-1, 0]])
        points2 = np.dot(points1, rotation)

        n = _difference_norm(points1, points2, embedding="ase", test_case="rotation")
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

        n = _difference_norm(
            points1, points2, embedding="ase", test_case="diagonal-rotation"
        )
        self.assertAlmostEqual(n, 0)

    def test_scalar_rotation_norm(self):
        # triangle in 2d
        points1 = np.array([[0, 0], [3, 0], [3, -2]], dtype=np.float64)
        rotation = np.array([[0, 1], [-1, 0]])
        # rotated 90 degrees
        points2 = np.dot(points1, rotation)
        # scaled
        points2 = 2 * points2

        n = _difference_norm(
            points1, points2, embedding="ase", test_case="scalar-rotation"
        )
        self.assertAlmostEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
