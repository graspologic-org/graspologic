import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from graspy.models import *
from graspy.simulations import er_np, sbm, sample_edges, p_from_latent
from graspy.utils import cartprod
from sklearn.metrics import adjusted_rand_score


class TestER:
    @classmethod
    def setup_class(cls):
        np.random.seed(8888)
        cls.graph = er_np(1000, 0.5)
        cls.p = 0.5
        cls.estimator = EREstimator(directed=True, loops=False)
        cls.estimator.fit(cls.graph)
        cls.p_hat = cls.estimator.p_

    def test_ER_inputs(self):
        ere = EREstimator()

        with pytest.raises(TypeError):
            EREstimator(directed="hey")

        with pytest.raises(TypeError):
            EREstimator(loops=6)

        with pytest.raises(ValueError):
            ere.fit(self.graph[:, :99])

        with pytest.raises(ValueError):
            ere.fit(self.graph[..., np.newaxis])

    def test_ER_fit(self):
        assert self.p_hat - self.p < 0.001

    def test_ER_sample(self):
        with pytest.raises(ValueError):
            self.estimator.sample(n_samples=-1)

        with pytest.raises(TypeError):
            self.estimator.sample(n_samples="nope")


class TestSBM:
    def test_SBM_inputs(self):
        with pytest.raises(TypeError):
            SBEstimator(directed="hey")

        with pytest.raises(TypeError):
            SBEstimator(loops=6)

        with pytest.raises(TypeError):
            SBEstimator(n_components="XD")

        with pytest.raises(ValueError):
            SBEstimator(n_components=-1)

        with pytest.raises(TypeError):
            SBEstimator(min_comm="1")

        with pytest.raises(ValueError):
            SBEstimator(min_comm=-1)

        with pytest.raises(TypeError):
            SBEstimator(max_comm="ay")

        with pytest.raises(ValueError):
            SBEstimator(max_comm=-1)

        with pytest.raises(ValueError):
            SBEstimator(min_comm=4, max_comm=2)

        graph = er_np(100, 0.5)
        bad_y = np.zeros(99)
        sbe = SBEstimator()
        with pytest.raises(ValueError):
            sbe.fit(graph, y=bad_y)

        with pytest.raises(ValueError):
            sbe.fit(graph[:, :99])

        with pytest.raises(ValueError):
            sbe.fit(graph[..., np.newaxis])

        with pytest.raises(TypeError):
            SBEstimator(cluster_kws=1)

        with pytest.raises(TypeError):
            SBEstimator(embed_kws=1)

    def test_SBM_fit_supervised(self):
        np.random.seed(8888)
        B = np.array(
            [
                [0.9, 0.2, 0.05, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.2, 0.4, 0.8, 0.5],
                [0.1, 0.2, 0.1, 0.7],
            ]
        )
        n = np.array([500, 500, 250, 250])
        g = sbm(n, B, directed=True, loops=False)
        sbe = SBEstimator(directed=True, loops=False)
        labels = _n_to_labels(n)
        sbe.fit(g, y=labels)
        B_hat = sbe.block_p_
        assert_allclose(B_hat, B, atol=0.01)

    def test_SBM_fit_unsupervised(self):
        np.random.seed(12345)
        n_verts = 1500

        B = np.array([[0.7, 0.1, 0.1], [0.1, 0.9, 0.1], [0.05, 0.1, 0.75]])
        n = np.array([500, 500, 500])
        labels = _n_to_labels(n)
        p_mat = _block_to_full(B, labels, (n_verts, n_verts))
        p_mat -= np.diag(np.diag(p_mat))
        graph = sample_edges(p_mat, directed=True, loops=False)
        sbe = SBEstimator(directed=True, loops=False)
        sbe.fit(graph)
        assert adjusted_rand_score(labels, sbe.vertex_assignments_) > 0.95
        assert_allclose(p_mat, sbe.p_mat_, atol=0.12)


class TestDCSBM:
    @classmethod
    def setup_class(cls):
        np.random.seed(8888)
        B = np.array(
            [
                [0.9, 0.2, 0.05, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.2, 0.4, 0.8, 0.5],
                [0.1, 0.2, 0.1, 0.7],
            ]
        )
        n = np.array([1000, 1000, 500, 500])
        dc = np.random.beta(2, 5, size=n.sum())
        labels = _n_to_labels(n)
        p_mat = _block_to_full(B, labels, (n.sum(), n.sum()))
        p_mat = p_mat * np.outer(dc, dc)
        p_mat -= np.diag(np.diag(p_mat))
        g = sample_edges(p_mat, directed=True, loops=False)
        cls.p_mat = p_mat
        cls.labels = labels
        cls.g = g

    def test_DCSBM_sample(self):
        dcsbe = DCSBEstimator()
        dcsbe.fit(self.g)
        estimator = dcsbe

        with pytest.raises(ValueError):
            estimator.sample(n_samples=-1)

        with pytest.raises(TypeError):
            estimator.sample(n_samples="nope")

        graphs = dcsbe.sample(n_samples=20)
        assert graphs.shape == (20, 3000, 3000)
        # TODO worth checking for accuracy here too? 

    def test_DCSBM_score(self):
        # TODO 
        

    def test_DCSBM_score_samples(self):
        # TODO 

    def test_DCSBM_fit_supervised(self):
        p_mat = self.p_mat
        labels = self.labels
        g = self.g
        dcsbe = DCSBEstimator(directed=True, loops=False)
        dcsbe.fit(g, y=labels)
        assert_allclose(dcsbe.p_mat_, p_mat, atol=0.1)

    def test_DCSBM_inputs(self):
        with pytest.raises(TypeError):
            DCSBEstimator(directed="hey")

        with pytest.raises(TypeError):
            DCSBEstimator(loops=6)

        with pytest.raises(TypeError):
            DCSBEstimator(n_components="XD")

        with pytest.raises(ValueError):
            DCSBEstimator(n_components=-1)

        with pytest.raises(TypeError):
            DCSBEstimator(min_comm="1")

        with pytest.raises(ValueError):
            DCSBEstimator(min_comm=-1)

        with pytest.raises(TypeError):
            DCSBEstimator(max_comm="ay")

        with pytest.raises(ValueError):
            DCSBEstimator(max_comm=-1)

        with pytest.raises(ValueError):
            DCSBEstimator(min_comm=4, max_comm=2)

        graph = er_np(100, 0.5)
        bad_y = np.zeros(99)
        dcsbe = DCSBEstimator()
        with pytest.raises(ValueError):
            dcsbe.fit(graph, y=bad_y)

        with pytest.raises(ValueError):
            dcsbe.fit(graph[:, :99])

        with pytest.raises(ValueError):
            dcsbe.fit(graph[..., np.newaxis])

        with pytest.raises(TypeError):
            DCSBEstimator(cluster_kws=1)

        with pytest.raises(TypeError):
            DCSBEstimator(embed_kws=1)

    def test_DCSBM_fit_unsupervised(self):
        np.random.seed(12345)
        n_verts = 1500

        distances = np.random.beta(4, 1, n_verts)
        B = np.array([[0.7, 0.1, 0.1], [0.1, 0.9, 0.1], [0.05, 0.1, 0.75]])
        n = np.array([500, 500, 500])
        labels = _n_to_labels(n)
        p_mat = _block_to_full(B, labels, (n_verts, n_verts))
        p_mat = p_mat * np.outer(distances, distances)
        p_mat -= np.diag(np.diag(p_mat))
        graph = sample_edges(p_mat, directed=True, loops=False)
        dcsbe = DCSBEstimator(directed=True, loops=False)
        dcsbe.fit(graph)
        assert adjusted_rand_score(labels, dcsbe.vertex_assignments_) > 0.95
        assert_allclose(p_mat, dcsbe.p_mat_, atol=0.12)


def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


def _block_to_full(block_mat, inverse, shape):
    """
    "blows up" a k x k matrix, where k is the number of communities, 
    into a full n x n probability matrix

    block mat : k x k 
    inverse : array like length n, 
    """
    block_map = cartprod(inverse, inverse).T
    mat_by_edge = block_mat[block_map[0], block_map[1]]
    full_mat = mat_by_edge.reshape(shape)
    return full_mat
