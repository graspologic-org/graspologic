# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
from numpy.testing import assert_equal
from sklearn.exceptions import NotFittedError

from graspologic.cluster.kclust import KMeansCluster


class TestKMeansCluster(unittest.TestCase):
    def test_inputs(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(100, 3))

        with self.assertRaises(TypeError):
            max_clusters = "1"
            kclust = KMeansCluster(max_clusters=max_clusters)

        # max_cluster < 0
        with self.assertRaises(ValueError):
            kclust = KMeansCluster(max_clusters=-1)

        # max_cluster more than n_samples
        with self.assertRaises(ValueError):
            kclust = KMeansCluster(max_clusters=1000)
            kclust.fit_predict(X)

    def test_predict_without_fit(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(100, 3))

        with self.assertRaises(NotFittedError):
            kclust = KMeansCluster(max_clusters=2)
            kclust.predict(X)

    def test_outputs_gaussians(self):
        np.random.seed(2)

        n = 100
        d = 3
        num_sims = 10
        for _ in range(num_sims):
            X1 = np.random.normal(2, 0.5, size=(n, d))
            X2 = np.random.normal(-2, 0.5, size=(n, d))
            X = np.vstack((X1, X2))
            y = np.repeat([0, 1], n)

            kclust = KMeansCluster(max_clusters=5)
            kclust.fit(X, y)
            aris = kclust.ari_

            # Assert that the two cluster model is the best
            assert_equal(np.max(aris), 1)

    def test_no_y(self):
        np.random.seed(2)
        n = 100
        d = 3
        X1 = np.random.normal(2, 0.5, size=(n, d))
        X2 = np.random.normal(-2, 0.5, size=(n, d))
        X = np.vstack((X1, X2))

        kclust = KMeansCluster(max_clusters=5)
        kclust.fit(X)

        assert_equal(np.argmax(kclust.silhouette_), 0)
