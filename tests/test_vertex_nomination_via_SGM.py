# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np

from graspologic.nominate import VNviaSGM
from graspologic.simulations import er_np

np.random.seed(1)


class TestVNviaSGM(unittest.TestCase):
    def test_VNviaSGM_inputs(self):
        with self.assertRaises(ValueError):
            VNviaSGM(order_voi_subgraph=-1)
        with self.assertRaises(ValueError):
            VNviaSGM(order_voi_subgraph=1.5)
        with self.assertRaises(ValueError):
            VNviaSGM(order_seeds_subgraph=-1)
        with self.assertRaises(ValueError):
            VNviaSGM(order_seeds_subgraph=1.5)
        with self.assertRaises(ValueError):
            VNviaSGM(n_init=-1)
        with self.assertRaises(ValueError):
            VNviaSGM(n_init=1.5)
        with self.assertRaises(ValueError):
            VNviaSGM(max_nominations=0)

        with self.assertRaises(ValueError):
            VNviaSGM().fit(
                np.random.randn(3, 4),
                np.random.randn(4, 4),
                0,
                [np.arange(2), np.arange(2)],
            )
        with self.assertRaises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(3, 4),
                0,
                [np.arange(2), np.arange(2)],
            )
        with self.assertRaises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(4, 4),
                0,
                [np.arange(2), 1],
            )
        with self.assertRaises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(4, 4),
                0,
                np.random.randn(3, 3),
            )
        with self.assertRaises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(4, 4),
                0,
                [np.arange(2), np.arange(3)],
            )
        with self.assertRaises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(4, 4),
                0,
                [np.arange(5), np.arange(5)],
            )
        with self.assertRaises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(4, 4),
                0,
                [[], []],
            )
        with self.assertRaises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(4, 4),
                0,
                [[1, 1], [1, 2]],
            )
        with self.assertRaises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(4, 4),
                0,
                [[1, 5], [1, 2]],
            )

    def test_vn_algorithm(self):
        g1 = er_np(n=50, p=0.6)
        node_shuffle = np.random.permutation(50)

        g2 = g1[np.ix_(node_shuffle, node_shuffle)]

        kklst = [(xx, yy) for xx, yy in zip(node_shuffle, np.arange(len(node_shuffle)))]
        kklst.sort(key=lambda x: x[0])
        kklst = np.array(kklst)

        voi = 7
        nseeds = 6

        vnsgm = VNviaSGM()
        nomlst = vnsgm.fit_predict(
            g1, g2, voi, [kklst[0:nseeds, 0], kklst[0:nseeds, 1]]
        )

        self.assertEqual(nomlst[0][0], kklst[np.where(kklst[:, 0] == voi)[0][0], 1])
