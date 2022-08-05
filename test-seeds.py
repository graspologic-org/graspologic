# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np 
import time
import matplotlib.pyplot as plt
from graspologic.plot import heatmap
from scipy.stats import special_ortho_group, ortho_group

from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes, SeededProcrustes

class TestSeededProcrustes(unittest.TestCase):
    def test_aligning_datasets(self):
        np.random.seed(314) 
        X = np.random.uniform(0, 1, (10, 2)) 
        Q = special_ortho_group.rvs(2) 
        Y = X @ Q 
        inds = [1,0,3,4,2]
        Y[inds,:]
        seeds = np.array([[0,1,2,3,4],[1,0,3,4,2]]) 
        
        aligner_SP = SeededProcrustes() 
        X_prime_SP = aligner_SP.fit_transform(X, Y, seeds) 
        Q = aligner_SP.fit(X,Y,seeds).Q_
        self.assertTrue(np.linalg.norm(Y.mean(axis=0) - (X @ Q).mean(axis=0)) < 0.1)


