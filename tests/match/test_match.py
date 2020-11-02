# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pytest
import numpy as np
import math
import random
from graspologic.match import GraphMatch as GMP
from graspologic.simulations import er_np, sbm_corr
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.align import SignFlips

np.random.seed(0)


class TestGMP:
    @classmethod
    def setup_class(cls):
        cls.barycenter = GMP(gmp=False)
        cls.rand = GMP(n_init=100, init="rand", gmp=False)
        cls.barygm = GMP(gmp=True)

    def test_SGM_inputs(self):
        with pytest.raises(TypeError):
            GMP(n_init=-1.5)
        with pytest.raises(ValueError):
            GMP(init="random")
        with pytest.raises(TypeError):
            GMP(max_iter=-1.5)
        with pytest.raises(TypeError):
            GMP(shuffle_input="hey")
        with pytest.raises(TypeError):
            GMP(eps=-1)
        with pytest.raises(TypeError):
            GMP(gmp="hey")
        with pytest.raises(TypeError):
            GMP(padding=2)
        with pytest.raises(ValueError):
            GMP(padding="hey")
        with pytest.raises(ValueError):
            GMP().fit(
                np.random.random((3, 4)),
                np.random.random((3, 4)),
                np.arange(2),
                np.arange(2),
            )
        with pytest.raises(ValueError):
            GMP().fit(
                np.random.random((3, 4)),
                np.random.random((3, 4)),
                np.arange(2),
                np.arange(2),
            )
        with pytest.raises(ValueError):
            GMP().fit(np.identity(3), np.identity(3), np.identity(3), np.arange(2))
        with pytest.raises(ValueError):
            GMP().fit(np.identity(3), np.identity(3), np.arange(1), np.arange(2))
        with pytest.raises(ValueError):
            GMP().fit(np.identity(3), np.identity(3), np.arange(5), np.arange(5))
        with pytest.raises(ValueError):
            GMP().fit(
                np.identity(3), np.identity(3), -1 * np.arange(2), -1 * np.arange(2)
            )
        with pytest.raises(ValueError):
            GMP().fit(
                np.random.random((4, 4)),
                np.random.random((4, 4)),
                np.arange(2),
                np.arange(2),
                np.random.random((3, 4)),
            )
        with pytest.raises(ValueError):
            GMP().fit(
                np.random.random((4, 4)),
                np.random.random((4, 4)),
                np.arange(2),
                np.arange(2),
                np.random.random((3, 3)),
            )

    def _get_AB(self, qap_prob):
        with open("tests/match/qapdata/" + qap_prob + ".dat") as f:
            f = [int(elem) for elem in f.read().split()]

            # adjusting
            f = np.array(f[1:])
            n = int(math.sqrt(len(f) / 2))
            f = f.reshape(2 * n, n)
            A = f[:n, :]
            B = f[n:, :]
            return A, B

    def test_barycenter_SGM(self):

        A, B = self._get_AB("chr12c")
        n = A.shape[0]
        pi = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n
        W1 = [4, 8, 10]
        W2 = [pi[z] for z in W1]
        chr12c = self.barycenter.fit(A, B, W1, W2)
        score = chr12c.score_
        assert 11156 <= score < 21000

        W1 = np.sort(random.sample(list(range(n)), n - 1))
        W2 = [pi[z] for z in W1]
        chr12c = self.barycenter.fit(A, B, W1, W2)
        score = chr12c.score_
        assert 11156 == score

    def test_rand_SGM(self):
        A, B = self._get_AB("chr12c")
        chr12c = self.rand.fit(A, B)
        score = chr12c.score_
        assert 11156 <= score < 13500

        n = A.shape[0]
        pi = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n
        W1 = [4, 8, 10]
        W2 = [pi[z] for z in W1]
        chr12c = self.rand.fit(A, B, W1, W2)
        score = chr12c.score_
        assert 11156 <= score < 12500

    def test_padding(self):
        n = 50
        p = 0.4

        np.random.seed(1)
        G1 = er_np(n=n, p=p)
        G2 = G1[: (n - 1), : (n - 1)]  # remove two nodes
        gmp_adopted = GMP(padding="adopted")
        res = gmp_adopted.fit(G1, G2)

        assert 1.0 == (sum(res.perm_inds_ == np.arange(n)) / n)

    def test_sim(self):
        n = 150
        rho = 1.0
        n_per_block = int(n / 3)
        n_blocks = 3
        block_members = np.array(n_blocks * [n_per_block])
        block_probs = np.array(
            [[0.2, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.2]]
        )
        directed = False
        loops = False
        A1, A2 = sbm_corr(
            block_members, block_probs, rho, directed=directed, loops=loops
        )
        ase = AdjacencySpectralEmbed(n_components=3, algorithm="truncated")
        x1 = ase.fit_transform(A1)
        x2 = ase.fit_transform(A2)
        xh1 = SignFlips().fit_transform(x1, x2)
        S = xh1 @ x2.T
        res = self.barygm.fit(A1, A2, S=S)

        assert 1.0 == (sum(res.perm_inds_ == np.arange(n)) / n)
