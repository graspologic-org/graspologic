# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
from scipy import stats
from scipy.stats import special_ortho_group

from graspologic.align import OrthogonalProcrustes, SeededProcrustes, SeedlessProcrustes


class TestSeededProcrustes(unittest.TestCase):
    def test_aligning_datasets(self):
        aligner = SeededProcrustes()
        X = np.random.uniform(0, 1, (20, 2))
        Q = special_ortho_group.rvs(2)
        Y = np.random.uniform(0, 1, (30, 2)) @ Q
        X_prime = np.random.uniform(0, 1, (10, 2))
        Y_prime = X_prime @ Q
        X_hat = np.vstack((X_prime, X))
        Y_hat = np.vstack((Y_prime, Y))
        X_prime_len = X_prime.shape[0]
        seeds = np.arange(X_prime_len)
        seeds = np.vstack((seeds, seeds)).transpose()
        aligner.fit_transform(X_hat, Y_hat, seeds)
