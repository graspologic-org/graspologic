# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from sklearn.exceptions import NotFittedError

from graspologic.cluster.autogmm import AutoGMMCluster
from graspologic.embed.ase import AdjacencySpectralEmbed
from graspologic.simulations.simulations import sbm


class TestAutoGMM(unittest.TestCase):
    def test_inputs(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(10, 3))

        # min_components < 1
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(min_components=0)

        # min_components integer
        with self.assertRaises(TypeError):
            AutoGMM = AutoGMMCluster(min_components="1")

        # max_components < min_components
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(min_components=1, max_components=0)

        # max_components integer
        with self.assertRaises(TypeError):
            AutoGMM = AutoGMMCluster(min_components=1, max_components="1")

        # affinity is not an array, string or list
        with self.assertRaises(TypeError):
            AutoGMM = AutoGMMCluster(min_components=1, affinity=1)

        # affinity is not in ['euclidean', 'manhattan', 'cosine', 'none']
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(min_components=1, affinity="graspologic")

        # linkage is not an array, string or list
        with self.assertRaises(TypeError):
            AutoGMM = AutoGMMCluster(min_components=1, linkage=1)

        # linkage is not in ['single', 'average', 'complete', 'ward']
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(min_components=1, linkage="graspologic")

        # euclidean is not an affinity option when ward is a linkage option
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(
                min_components=1, affinity="manhattan", linkage="ward"
            )

        # covariance type is not an array, string or list
        with self.assertRaises(TypeError):
            AutoGMM = AutoGMMCluster(min_components=1, covariance_type=1)

        # covariance type is not in ['spherical', 'diag', 'tied', 'full']
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(min_components=1, covariance_type="graspologic")

        # min_cluster > n_samples when max_cluster is None
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(1000)
            AutoGMM.fit(X)

        # max_cluster > n_samples when max_cluster is not None
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(10, 1001)
            AutoGMM.fit(X)

        # min_cluster > n_samples when max_cluster is None
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(1000)
            AutoGMM.fit(X)

        # min_cluster > n_samples when max_cluster is not None
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(1000, 1001)
            AutoGMM.fit(X)

        # label_init is not a 1-D array
        with self.assertRaises(TypeError):
            AutoGMM = AutoGMMCluster(label_init=np.zeros([10, 2]))

        # label_init is not 1-D array, a list or None.
        with self.assertRaises(TypeError):
            AutoGMM = AutoGMMCluster(label_init="label")

        # label_init length is not equal to n_samples
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(label_init=np.zeros([5, 1]))
            AutoGMM.fit(X)

        # criter = cic
        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(selection_criteria="cic")

    def test_labels_init(self):
        X = np.random.normal(0, 1, size=(5, 3))

        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(
                min_components=1, max_components=1, label_init=np.array([0, 0, 0, 0, 1])
            )
            AutoGMM.fit_predict(X)

        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(
                min_components=1, max_components=2, label_init=np.array([0, 0, 0, 0, 1])
            )
            AutoGMM.fit_predict(X)

        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(
                min_components=2, max_components=3, label_init=np.array([0, 0, 0, 0, 1])
            )
            AutoGMM.fit_predict(X)

        AutoGMM = AutoGMMCluster(
            min_components=2, max_components=2, label_init=np.array([0, 0, 0, 0, 1])
        )
        AutoGMM.fit_predict(X)

    def test_predict_without_fit(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(10, 3))

        with self.assertRaises(NotFittedError):
            AutoGMM = AutoGMMCluster(min_components=2)
            AutoGMM.predict(X)

    def test_cosine_on_0(self):
        X = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 0, 1]])

        with self.assertRaises(ValueError):
            AutoGMM = AutoGMMCluster(min_components=3, affinity="all")
            AutoGMM.fit(X)

    def test_no_y(self):
        np.random.seed(1)

        n = 10
        d = 1

        X1 = np.random.normal(10, 0.5, size=(n, d))
        X2 = np.random.normal(-10, 0.5, size=(n, d))
        X = np.vstack((X1, X2))

        AutoGMM = AutoGMMCluster(max_components=5)
        AutoGMM.fit(X)

        assert_equal(AutoGMM.n_components_, 2)

    def test_two_class(self):
        """
        Easily separable two gaussian problem.
        """
        np.random.seed(1)

        n = 10
        d = 1

        X1 = np.random.normal(10, 0.5, size=(n, d))
        X2 = np.random.normal(-10, 0.5, size=(n, d))
        X = np.vstack((X1, X2))
        y = np.repeat([0, 1], n)

        AutoGMM = AutoGMMCluster(max_components=5)
        AutoGMM.fit(X, y)

        n_components = AutoGMM.n_components_

        # Assert that the two cluster model is the best
        assert_equal(n_components, 2)

        # Asser that we get perfect clustering
        assert_allclose(AutoGMM.ari_, 1)
