# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.utils.estimator_checks import check_estimator

from graspologic.embed.mds import ClassicalMDS


class TestMDS(unittest.TestCase):
    def test_sklearn_conventions(self):
        check_estimator(ClassicalMDS())

    def test_input(self):
        X = np.random.normal(0, 1, size=(10, 3))

        # X cannot be tensor when precomputed dissimilarity
        with self.assertRaises(ValueError):
            tensor = np.random.normal(0, 1, size=(10, 3, 3))
            mds = ClassicalMDS(n_components=3, dissimilarity="precomputed")
            mds.fit(tensor)

        with self.assertRaises(ValueError):
            one_dimensional = np.random.normal(size=10)
            mds = ClassicalMDS(n_components=2, dissimilarity="euclidean")
            mds.fit(one_dimensional)

        # n_components > n_samples
        with self.assertRaises(ValueError):
            mds = ClassicalMDS(n_components=100)
            mds.fit(X)

        # Invalid n_components
        with self.assertRaises(ValueError):
            mds = ClassicalMDS(n_components=-2)

        with self.assertRaises(TypeError):
            mds = ClassicalMDS(n_components="1")

        # Invalid dissimilarity
        with self.assertRaises(ValueError):
            mds = ClassicalMDS(dissimilarity="abc")

        # Invalid input for fit function
        with self.assertRaises(ValueError):
            mds = ClassicalMDS(n_components=3, dissimilarity="precomputed")
            mds.fit(X="bad_input")

        # Must be square and symmetric matrix if precomputed dissimilarity
        with self.assertRaises(ValueError):
            mds = ClassicalMDS(n_components=3, dissimilarity="precomputed")
            mds.fit(X)

    def test_tensor_input(self):
        X = np.random.normal(size=(100, 5, 5))
        mds = ClassicalMDS(n_components=3, dissimilarity="euclidean")
        mds.fit(X)

        self.assertEqual(mds.dissimilarity_matrix_.shape, (100, 100))

        X_transformed = mds.fit_transform(X)
        self.assertEqual(X_transformed.shape, (100, 3))

    def test_output(self):
        """
        Recover a 3D tetrahedron with distance 1 between all points

        Use both fit and fit_transform functions
        """

        def _compute_dissimilarity(arr):
            out = np.zeros((4, 4))
            for i in range(4):
                out[i] = np.linalg.norm(arr - arr[i], axis=1)

            return out

        def use_fit_transform():
            A = np.ones((4, 4)) - np.identity(4)

            mds = ClassicalMDS(n_components=3, dissimilarity="precomputed")
            B = mds.fit_transform(A)

            Ahat = _compute_dissimilarity(B)

            # Checks up to 7 decimal points
            assert_almost_equal(A, Ahat)

        def use_fit():
            A = np.ones((4, 4)) - np.identity(4)

            mds = ClassicalMDS(n_components=3, dissimilarity="precomputed")
            mds.fit(A)
            B = np.dot(mds.components_, np.diag(mds.singular_values_))

            Ahat = _compute_dissimilarity(B)

            # Checks up to 7 decimal points
            assert_almost_equal(A, Ahat)

        def use_euclidean():
            A = np.array([
                [-7.62291243e-17, 6.12372436e-01, 4.95031815e-16],
                [-4.97243701e-01, -2.04124145e-01, -2.93397401e-01],
                [5.02711453e-01, -2.04124145e-01, -2.83926977e-01],
                [-5.46775198e-03, -2.04124145e-01, 5.77324378e-01],
            ])

            mds = ClassicalMDS(dissimilarity="euclidean")
            B = mds.fit_transform(A)

            target = np.ones((4, 4)) - np.identity(4)
            assert_almost_equal(mds.dissimilarity_matrix_, target)

        use_fit_transform()
        use_fit()
        use_euclidean()
