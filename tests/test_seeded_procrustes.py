# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np 
from scipy import stats
from scipy.stats import special_ortho_group

from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes, SeededProcrustes

class TestSeededProcrustes(unittest.TestCase):
    def test_aligning_datasets(self):
        np.random.seed(314) 
        X = np.random.uniform(0, 1, (10, 2)) 
        Q_ = special_ortho_group.rvs(2) 
        Y = X @ Q_
        inds = [1,0,3,4,2]
        Y[inds,:]
        seeds = np.array([[0,1,2,3,4],inds]) 
        aligner_SP = SeededProcrustes() 
        aligner_SP.fit_transform(X, Y, seeds) 
        Q_ = aligner_SP.fit(X,Y,seeds).Q_
        self.assertTrue(np.linalg.norm(Y.mean(axis=0) - (X @ Q_).mean(axis=0)) < 0.1)

        aligner = SeededProcrustes()
        X = np.random.uniform(0, 1, (20, 2))
        Q = special_ortho_group.rvs(2)
        Y = np.random.uniform(0, 1, (30, 2)) @ Q
        X_prime = np.random.uniform(0, 1, (10, 2))
        Y_prime = X_prime @ Q
        X_hat = np.vstack((X_prime,X))
        Y_hat = np.vstack((Y_prime,Y))
        X_prime_len = X_prime.shape[0]
        seeds = np.arange(X_prime_len)


