# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from sklearn.exceptions import NotFittedError

from graspologic.cluster.gclust import GaussianCluster
from graspologic.embed.ase import AdjacencySpectralEmbed
from graspologic.simulations.simulations import sbm


class TestGaussianCluster(unittest.TestCase):
    def test_inputs(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(100, 3))

        # min_components < 1
        with self.assertRaises(ValueError):
            gclust = GaussianCluster(min_components=0)

        # min_components integer
        with self.assertRaises(TypeError):
            gclust = GaussianCluster(min_components="1")

        # max_components < min_components
        with self.assertRaises(ValueError):
            gclust = GaussianCluster(min_components=1, max_components=0)

        # max_components integer
        with self.assertRaises(TypeError):
            gclust = GaussianCluster(min_components=1, max_components="1")

        # covariance type is not an array, string or list
        with self.assertRaises(TypeError):
            gclust = GaussianCluster(min_components=1, covariance_type=1)

        # covariance type is not in ['spherical', 'diag', 'tied', 'full']
        with self.assertRaises(ValueError):
            gclust = GaussianCluster(min_components=1, covariance_type="graspologic")

        # min_cluster > n_samples when max_cluster is None
        with self.assertRaises(ValueError):
            gclust = GaussianCluster(1000)
            gclust.fit(X)

        with self.assertRaises(ValueError):
            gclust = GaussianCluster(1000)
            gclust.fit_predict(X)

        # max_cluster > n_samples when max_cluster is not None
        with self.assertRaises(ValueError):
            gclust = GaussianCluster(10, 1001)
            gclust.fit(X)

        with self.assertRaises(ValueError):
            gclust = GaussianCluster(10, 1001)
            gclust.fit_predict(X)

        # min_cluster > n_samples when max_cluster is None
        with self.assertRaises(ValueError):
            gclust = GaussianCluster(1000)
            gclust.fit(X)

        with self.assertRaises(ValueError):
            gclust = GaussianCluster(10, 1001)
            gclust.fit_predict(X)

        # min_cluster > n_samples when max_cluster is not None
        with self.assertRaises(ValueError):
            gclust = GaussianCluster(1000, 1001)
            gclust.fit(X)

        with self.assertRaises(ValueError):
            gclust = GaussianCluster(1000, 1001)
            gclust.fit_predict(X)

    def test_predict_without_fit(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(100, 3))

        with self.assertRaises(NotFittedError):
            gclust = GaussianCluster(min_components=2)
            gclust.predict(X)

    def test_no_y(self):
        np.random.seed(2)

        n = 100
        d = 3

        X1 = np.random.normal(2, 0.5, size=(n, d))
        X2 = np.random.normal(-2, 0.5, size=(n, d))
        X = np.vstack((X1, X2))

        gclust = GaussianCluster(min_components=5, n_init=2)
        gclust.fit(X)

        assert_equal(gclust.n_components_, 2)

    def test_two_class(self):
        """
        Easily separable two gaussian problem.
        """
        np.random.seed(2)

        n = 100
        d = 3

        num_sims = 10

        for _ in range(num_sims):
            X1 = np.random.normal(2, 0.5, size=(n, d))
            X2 = np.random.normal(-2, 0.5, size=(n, d))
            X = np.vstack((X1, X2))
            y = np.repeat([0, 1], n)

            gclust = GaussianCluster(min_components=5)
            gclust.fit(X, y)

            n_components = gclust.n_components_

            # Assert that the two cluster model is the best
            assert_equal(n_components, 2)

            # Asser that we get perfect clustering
            assert_allclose(gclust.ari_.loc[n_components], 1)

    def test_five_class(self):
        """
        Easily separable five gaussian problem.
        """
        np.random.seed(10)

        n = 100
        mus = [[i * 5, 0] for i in range(5)]
        cov = np.eye(2)  # balls

        num_sims = 10

        for _ in range(num_sims):
            X = np.vstack([np.random.multivariate_normal(mu, cov, n) for mu in mus])

            gclust = GaussianCluster(
                min_components=3, max_components=10, covariance_type="all"
            )
            gclust.fit(X)
            assert_equal(gclust.n_components_, 5)

    def test_ase_three_blocks(self):
        """
        Expect 3 clusters from a 3 block model
        """
        np.random.seed(3)
        num_sims = 10

        # Generate adjacency and labels
        n = 50
        n_communites = [n, n, n]
        p = np.array([[0.8, 0.3, 0.2], [0.3, 0.8, 0.3], [0.2, 0.3, 0.8]])
        y = np.repeat([1, 2, 3], repeats=n)

        for _ in range(num_sims):
            A = sbm(n=n_communites, p=p)

            # Embed to get latent positions
            ase = AdjacencySpectralEmbed(n_components=5)
            X_hat = ase.fit_transform(A)

            # Compute clusters
            gclust = GaussianCluster(min_components=10)
            gclust.fit(X_hat, y)

            n_components = gclust.n_components_

            # Assert that the three cluster model is the best
            assert_equal(n_components, 3)

            # Asser that we get perfect clustering
            assert_allclose(gclust.ari_.loc[n_components], 1)

    def test_covariances(self):
        """
        Easily separable two gaussian problem.
        """
        np.random.seed(2)

        n = 100

        mu1 = [-10, 0]
        mu2 = [10, 0]

        # Spherical
        cov1 = 2 * np.eye(2)
        cov2 = 2 * np.eye(2)

        X1 = np.random.multivariate_normal(mu1, cov1, n)
        X2 = np.random.multivariate_normal(mu2, cov2, n)

        X = np.concatenate((X1, X2))

        gclust = GaussianCluster(min_components=2, covariance_type="all")
        gclust.fit(X)
        assert_equal(gclust.covariance_type_, "spherical")

        # Diagonal
        np.random.seed(10)
        cov1 = np.diag([1, 1])
        cov2 = np.diag([2, 1])

        X1 = np.random.multivariate_normal(mu1, cov1, n)
        X2 = np.random.multivariate_normal(mu2, cov2, n)

        X = np.concatenate((X1, X2))

        gclust = GaussianCluster(min_components=2, covariance_type="all")
        gclust.fit(X)
        assert_equal(gclust.covariance_type_, "diag")

        # Tied
        cov1 = np.array([[2, 1], [1, 2]])
        cov2 = np.array([[2, 1], [1, 2]])

        X1 = np.random.multivariate_normal(mu1, cov1, n)
        X2 = np.random.multivariate_normal(mu2, cov2, n)

        X = np.concatenate((X1, X2))

        gclust = GaussianCluster(min_components=2, covariance_type="all")
        gclust.fit(X)
        assert_equal(gclust.covariance_type_, "tied")

        # Full
        cov1 = np.array([[2, -1], [-1, 2]])
        cov2 = np.array([[2, 1], [1, 2]])

        X1 = np.random.multivariate_normal(mu1, cov1, n)
        X2 = np.random.multivariate_normal(mu2, cov2, n)

        X = np.concatenate((X1, X2))

        gclust = GaussianCluster(min_components=2, covariance_type="all")
        gclust.fit(X)
        assert_equal(gclust.covariance_type_, "full")
