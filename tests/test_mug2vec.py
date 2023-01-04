# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np

from graspologic.cluster import GaussianCluster
from graspologic.embed import mug2vec
from graspologic.simulations import sbm


def generate_data():
    np.random.seed(2)

    p1 = [[0.3, 0.1], [0.1, 0.3]]
    p2 = [[0.1, 0.3], [0.3, 0.1]]
    n = [50, 50]

    g1 = [sbm(n, p1) for _ in range(20)]
    g2 = [sbm(n, p2) for _ in range(20)]
    g = g1 + g2

    y = ["0"] * 20 + ["1"] * 20

    return g, y


class TestMug2Vec(unittest.TestCase):
    def test_mug2vec(self):
        graphs, labels = generate_data()

        mugs = mug2vec(pass_to_ranks=None, svd_seed=1)
        xhat = mugs.fit_transform(graphs)

        gmm = GaussianCluster(5)
        gmm.fit(xhat, labels)

        self.assertEqual(gmm.n_components_, 2)

    def test_inputs(self):
        graphs, labels = generate_data()

        mugs = mug2vec(omnibus_components=-1, svd_seed=1)
        with self.assertRaises(ValueError):
            mugs.fit(graphs)

        mugs = mug2vec(cmds_components=-1, svd_seed=1)
        with self.assertRaises(ValueError):
            mugs.fit(graphs)

        mugs = mug2vec(omnibus_n_elbows=-1, svd_seed=1)
        with self.assertRaises(ValueError):
            mugs.fit(graphs)

        mugs = mug2vec(cmds_n_elbows=-1, svd_seed=1)
        with self.assertRaises(ValueError):
            mugs.fit(graphs)
