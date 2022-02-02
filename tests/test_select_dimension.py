# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
from numpy.testing import assert_equal
from scipy.linalg import orth

from graspologic.embed.svd import select_dimension
from graspologic.simulations.simulations import sbm


def generate_data(n=10, elbows=3, seed=1):
    """
    Generate data matrix with a specific number of elbows on scree plot
    """
    np.random.seed(seed)
    x = np.random.binomial(1, 0.6, (n**2)).reshape(n, n)
    xorth = orth(x)
    d = np.zeros(xorth.shape[0])
    for i in range(0, len(d), int(len(d) / (elbows + 1))):
        d[:i] += 10
    A = xorth.T.dot(np.diag(d)).dot(xorth)
    return A, d


class TestSelectDimension(unittest.TestCase):
    def test_invalid_inputes(self):
        X, D = generate_data()

        # invalid n_elbows
        with self.assertRaises(ValueError):
            bad_n_elbows = -2
            select_dimension(X, n_elbows=bad_n_elbows)

        with self.assertRaises(ValueError):
            bad_n_elbows = "string"
            select_dimension(X, n_elbows=bad_n_elbows)

        # invalid n_components
        with self.assertRaises(ValueError):
            bad_n_components = -1
            select_dimension(X, n_components=bad_n_components)

        with self.assertRaises(ValueError):
            bad_n_components = "string"
            select_dimension(X, n_components=bad_n_components)

        # invalid threshold
        with self.assertRaises(ValueError):
            bad_threshold = -2
            select_dimension(X, threshold=bad_threshold)

        with self.assertRaises(ValueError):
            bad_threshold = "string"
            select_dimension(X, threshold=bad_threshold)

        with self.assertRaises(IndexError):
            bad_threshold = 1000000
            select_dimension(X, threshold=bad_threshold)

        # invalid X
        with self.assertRaises(ValueError):
            bad_X = -2
            select_dimension(X=bad_X)

        with self.assertRaises(ValueError):
            # input is tensor
            bad_X = np.random.normal(size=(100, 10, 10))
            select_dimension(X=bad_X)

        with self.assertRaises(ValueError):
            bad_X = np.random.normal(size=100).reshape(100, -1)
            select_dimension(X=bad_X)

    def test_output_synthetic(self):
        data, l = generate_data(10, 3)
        elbows, _, _ = select_dimension(X=data, n_elbows=2, return_likelihoods=True)
        assert_equal(elbows, [2, 4])

    def test_output_simple(self):
        """
        Elbow should be at 2.
        """
        X = np.array([10, 9, 3, 2, 1])
        elbows, _ = select_dimension(X, n_elbows=1)
        assert_equal(elbows[0], 2)

    def test_output_uniform(self):
        """
        Generate two sets of synthetic eigenvalues based on two uniform distributions.
        The elbow must be at 50.
        """
        np.random.seed(9)
        x1 = np.random.uniform(0, 45, 50)
        x2 = np.random.uniform(55, 100, 50)
        X = np.sort(np.hstack([x1, x2]))[::-1]
        elbows, _ = select_dimension(X, n_elbows=1)
        assert_equal(elbows[0], 50)

    def test_output_two_block_sbm(self):
        np.random.seed(10)
        n_communities = [100, 100]
        P = np.array([[0.5, 0.1], [0.1, 0.5]])
        A = sbm(n_communities, P)

        elbows, _ = select_dimension(A, n_elbows=2)
        assert_equal(elbows[0], 2)
