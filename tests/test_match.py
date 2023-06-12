# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import random
import unittest

import numpy as np
from beartype.roar import BeartypeCallHintParamViolation

from graspologic.match import graph_match
from graspologic.simulations import er_corr, er_np

np.random.seed(1)

# adjacency matrices from QAPLIB instance chr12c
# QAP problem is minimized with objective function value 11156
A = [
    [0, 90, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [90, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
    [10, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0],
    [0, 23, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0],
    [0, 0, 43, 0, 0, 0, 26, 0, 0, 0, 0, 0],
    [0, 0, 0, 88, 0, 0, 0, 16, 0, 0, 0, 0],
    [0, 0, 0, 0, 26, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 16, 0, 0, 0, 96, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 29, 0],
    [0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 37],
    [0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0],
]
B = [
    [0, 36, 54, 26, 59, 72, 9, 34, 79, 17, 46, 95],
    [36, 0, 73, 35, 90, 58, 30, 78, 35, 44, 79, 36],
    [54, 73, 0, 21, 10, 97, 58, 66, 69, 61, 54, 63],
    [26, 35, 21, 0, 93, 12, 46, 40, 37, 48, 68, 85],
    [59, 90, 10, 93, 0, 64, 5, 29, 76, 16, 5, 76],
    [72, 58, 97, 12, 64, 0, 96, 55, 38, 54, 0, 34],
    [9, 30, 58, 46, 5, 96, 0, 83, 35, 11, 56, 37],
    [34, 78, 66, 40, 29, 55, 83, 0, 44, 12, 15, 80],
    [79, 35, 69, 37, 76, 38, 35, 44, 0, 64, 39, 33],
    [17, 44, 61, 48, 16, 54, 11, 12, 64, 0, 70, 86],
    [46, 79, 54, 68, 5, 0, 56, 15, 39, 70, 0, 18],
    [95, 36, 63, 85, 76, 34, 37, 80, 33, 86, 18, 0],
]
A, B = np.array(A), np.array(B)


class TestGraphMatch(unittest.TestCase):
    def test_SGM_inputs(self):
        with self.assertRaises(BeartypeCallHintParamViolation):
            graph_match(A, B, n_init=-1.5)
        with self.assertRaises(BeartypeCallHintParamViolation):
            graph_match(A, B, init="not correct string")
        with self.assertRaises(BeartypeCallHintParamViolation):
            graph_match(A, B, max_iter=-1.5)
        with self.assertRaises(BeartypeCallHintParamViolation):
            graph_match(A, B, shuffle_input="hey")
        with self.assertRaises(ValueError):
            graph_match(A, B, tol=-1)
        with self.assertRaises(BeartypeCallHintParamViolation):
            graph_match(A, B, maximize="hey")
        with self.assertRaises(BeartypeCallHintParamViolation):
            graph_match(A, B, padding="hey")
        with self.assertRaises(ValueError):
            # A, B need to be square
            graph_match(
                np.random.random((3, 4)),
                np.random.random((3, 4)),
            )
        with self.assertRaises(ValueError):
            # BA, AB need to match A, B on certain dims
            graph_match(
                np.random.random((4, 4)),
                np.random.random((4, 4)),
                np.random.random((3, 3)),
                np.random.random((3, 3)),
            )
        with self.assertRaises(ValueError):
            # can't have more seeds than nodes
            graph_match(
                np.identity(3), np.identity(3), partial_match=np.full((5, 2), 1)
            )
        with self.assertRaises(ValueError):
            # can't have seeds that are smaller than 0
            graph_match(
                np.identity(3), np.identity(3), partial_match=np.full((2, 2), -1)
            )
        with self.assertRaises(ValueError):
            # size of similarity must fit with A, B
            graph_match(np.identity(3), np.identity(3), S=np.identity(4))

    def test_barycenter_SGM(self):
        # minimize such that we achieve some number close to the optimum,
        # though strictly greater than or equal
        # results vary due to random shuffle within GraphMatch

        n = A.shape[0]
        pi = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n
        seeds1 = [4, 8, 10]
        seeds2 = [pi[z] for z in seeds1]
        partial_match = np.column_stack((seeds1, seeds2))
        _, _, score, _ = graph_match(A, B, partial_match=partial_match, maximize=False)
        self.assertTrue(11156 <= score < 21000)

        seeds1 = np.sort(random.sample(list(range(n)), n - 1))
        seeds2 = [pi[z] for z in seeds1]
        partial_match = np.column_stack((seeds1, seeds2))
        _, _, score, _ = graph_match(A, B, partial_match=partial_match, maximize=False)
        self.assertEqual(11156, score)

        seeds1 = np.array(range(n))
        seeds2 = pi
        partial_match = np.column_stack((seeds1, seeds2))
        _, indices_B, score, _ = graph_match(
            A, B, partial_match=partial_match, maximize=False
        )
        np.testing.assert_array_equal(indices_B, pi)
        self.assertTrue(11156, score)

        seeds1 = np.random.permutation(n)
        seeds2 = [pi[z] for z in seeds1]
        partial_match = np.column_stack((seeds1, seeds2))
        _, indices_B, score, _ = graph_match(
            A, B, partial_match=partial_match, maximize=False
        )
        np.testing.assert_array_equal(indices_B, pi)
        self.assertTrue(11156, score)

    def test_barycenter_SGM_seed_lists(self):
        # minimize such that we achieve some number close to the optimum,
        # though strictly greater than or equal
        # results vary due to random shuffle within GraphMatch

        n = A.shape[0]
        pi = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n
        seeds1 = [4, 8, 10]
        seeds2 = [pi[z] for z in seeds1]
        _, _, score, _ = graph_match(
            A, B, partial_match=(seeds1, seeds2), maximize=False
        )
        self.assertTrue(11156 <= score < 21000)

    def test_parallel(self):
        _, _, score, _ = graph_match(
            A,
            B,
            maximize=False,
            n_init=2,
            n_jobs=2,
            rng=888,
        )
        self.assertTrue(11156 <= score < 13500)

    def test_padding(self):
        np.random.seed(888)
        n = 50
        p = 0.4

        A = er_np(n=n, p=p)
        B = A[:-2, :-2]  # remove two nodes

        indices_A, indices_B, _, _ = graph_match(A, B, rng=888, padding="adopted")

        self.assertTrue(np.array_equal(indices_A, np.arange(n - 2)))
        self.assertTrue(np.array_equal(indices_B, np.arange(n - 2)))

    def test_reproducibility(self):
        np.random.seed(888)
        n = 10
        p = 0.2
        A = er_np(n=n, p=p)
        B = A.copy()
        permutation = np.random.permutation(n)
        B = B[permutation][:, permutation]
        _, indices_B, _, _ = graph_match(A, B, rng=999)
        for i in range(10):
            # this fails w/o rng set here; i.e. there is variance
            _, indices_B_repeat, _, _ = graph_match(A, B, rng=999)
            self.assertTrue(np.array_equal(indices_B, indices_B_repeat))

    def test_custom_init(self):
        n = len(A)
        pi = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n
        custom_init = np.eye(n)
        custom_init = custom_init[pi]

        _, indices_B, score, _ = graph_match(A, B, init=custom_init, maximize=False)

        self.assertTrue((indices_B == pi).all())
        self.assertEqual(score, 11156)
        # we had thought about doing the test
        # `assert gm.n_iter_ == 1`
        # but note that GM doesn't necessarily converge in 1 iteration here
        # this is because when we start from the optimal permutation matrix, we do
        # not start from the optimal over our convex relaxation (the doubly stochastics)
        # but we do indeed recover the correct permutation after a small number of
        # iterations

    def test_wrong_custom_init(self):
        n = len(A)

        custom_init = np.full(n, 1 / n)
        with self.assertRaises(ValueError):
            _, indices_B, score, _ = graph_match(A, B, init=custom_init, maximize=False)

        custom_init = np.full((n, n), 1)
        with self.assertRaises(ValueError):
            _, indices_B, score, _ = graph_match(A, B, init=custom_init, maximize=False)

    def test_custom_init_seeds(self):
        n = len(A)
        pi_original = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - 1
        pi = np.array([5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - 1

        pi[pi > 6] -= 1

        # use seed 0 in A to 7 in B
        seeds_A = [0]
        seeds_B = [6]
        seeds = np.column_stack((seeds_A, seeds_B))
        custom_init = np.eye(n - 1)
        custom_init = custom_init[pi]

        # gm = GMP(n_init=1, init=custom_init, max_iter=30, shuffle_input=True, gmp=False)
        # gm.fit(A, B, seeds_A=seeds_A, seeds_B=seeds_B)
        _, indices_B, score, _ = graph_match(
            A,
            B,
            partial_match=seeds,
            n_init=1,
            init=custom_init,
            max_iter=30,
            maximize=False,
        )

        self.assertTrue((indices_B == pi_original).all())
        self.assertEqual(score, 11156)

    def test_similarity_term(self):
        rng = np.random.default_rng(888)
        np.random.seed(888)
        n = 10
        n_seeds = 1
        lamb = 0.8  # is diagnostic in the sense that w/ lamb=0, this test fails
        n_sims = 10
        mean_match_ratio = 0.0
        for _ in range(n_sims):
            A, B = er_corr(n, 0.3, 0.8, directed=True)
            perm = rng.permutation(n)
            undo_perm = np.argsort(perm)
            B = B[perm][:, perm]

            seeds_A = np.random.choice(n, replace=False, size=n_seeds)
            seeds_B = np.argsort(perm)[seeds_A]
            partial_match = np.stack((seeds_A, seeds_B)).T
            non_seeds_A = np.setdiff1d(np.arange(n), seeds_A)

            S = lamb * np.eye(B.shape[0])
            S = np.random.uniform(0, 1, (n, n)) + S
            S = S[:, perm]

            _, indices_B, _, _ = graph_match(A, B, S=S, partial_match=partial_match)
            mean_match_ratio += (
                indices_B[non_seeds_A] == undo_perm[non_seeds_A]
            ).mean() / n_sims

        self.assertTrue(mean_match_ratio >= 0.999)

    def test_similatiry_padded(self):
        np.random.seed(88)
        A = np.random.rand(10, 10)
        B = np.random.rand(11, 11)
        S = np.eye(10, 11) * 10

        _, perm_B, _, misc = graph_match(A, B, S=S)
        self.assertEqual((perm_B == np.arange(10)).mean(), 1.0)

        # want to make sure obj func value is the same as if we hadn't padded S,
        # should be since we use naive padding only
        score = misc[0]["score"]
        out_score = np.sum(A * B[perm_B][:, perm_B]) + np.trace(S[:, perm_B])
        self.assertEqual(score, out_score)
