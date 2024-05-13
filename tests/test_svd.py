# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
from scipy.spatial import procrustes

from graspologic.embed.svd import select_svd
from graspologic.utils import symmetrize


class TestSVD(unittest.TestCase):
    def test_bad_inputs(self):
        X = np.random.normal(size=(100, 5))
        with self.assertRaises(ValueError):
            bad_algo = "ROFLMAO"
            select_svd(X, algorithm=bad_algo)

        with self.assertRaises(ValueError):
            algorithm = "full"
            bad_components = 1000
            select_svd(X, n_components=bad_components, algorithm=algorithm)

        with self.assertRaises(ValueError):
            algorithm = "truncated"
            bad_components = 1000
            select_svd(X, n_components=bad_components, algorithm=algorithm)

    def test_outputs(self):
        np.random.seed(123)
        X = np.vstack([
            np.repeat([[0.2, 0.2, 0.2]], 50, axis=0),
            np.repeat([[0.5, 0.5, 0.5]], 50, axis=0),
        ])
        P = X @ X.T
        A = np.random.binomial(1, P).astype(float)

        n_components = 3

        # Full SVD
        U_full, D_full, V_full = select_svd(
            A, n_components=n_components, algorithm="full"
        )
        X_full = U_full @ np.diag(np.sqrt(D_full))
        _, _, norm_full = procrustes(X, X_full)

        # Truncated SVD
        U_trunc, D_trunc, V_trunc = select_svd(
            A, n_components=n_components, algorithm="truncated"
        )
        X_trunc = U_trunc @ np.diag(np.sqrt(D_trunc))
        _, _, norm_trunc = procrustes(X, X_trunc)

        # Randomized SVD
        U_rand, D_rand, V_rand = select_svd(
            A, n_components=n_components, algorithm="randomized", n_iter=10
        )
        X_rand = U_rand @ np.diag(np.sqrt(D_rand))
        _, _, norm_rand = procrustes(X, X_rand)

        rtol = 1e-4
        atol = 1e-4
        np.testing.assert_allclose(norm_full, norm_trunc, rtol, atol)
        np.testing.assert_allclose(norm_full, norm_rand, rtol, atol)

    def test_eigsh(self):
        np.random.seed(123)
        X = np.vstack([
            np.repeat([[0.2, 0.2, 0.2]], 50, axis=0),
            np.repeat([[0.5, 0.5, 0.5]], 50, axis=0),
        ])
        P = X @ X.T
        A = np.random.binomial(1, P).astype(float)
        A = symmetrize(A, method="triu")
        n_components = 3

        # Full SVD
        U_full, D_full, V_full = select_svd(
            A, n_components=n_components, algorithm="full"
        )
        X_full = U_full @ np.diag(np.sqrt(D_full))
        _, _, norm_full = procrustes(X, X_full)

        # eigsh SVD
        U_square, D_square, V_square = select_svd(
            A, n_components=n_components, algorithm="eigsh", n_iter=10
        )
        X_square = U_square @ np.diag(np.sqrt(D_square))
        _, _, norm_square = procrustes(X, X_square)

        rtol = 1e-4
        atol = 1e-4
        np.testing.assert_allclose(norm_full, norm_square, rtol, atol)
