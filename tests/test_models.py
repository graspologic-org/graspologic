import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from graspy.models import *
from graspy.simulations import er_np, sbm, sample_edges, p_from_latent


def test_ER_inputs():
    with pytest.raises(TypeError):
        EREstimator(directed="hey")

    with pytest.raises(TypeError):
        EREstimator(loops=6)

    graph = er_np(100, 0.5)
    ere = EREstimator()

    with pytest.raises(ValueError):
        ere.fit(graph[:, :99])

    with pytest.raises(ValueError):
        ere.fit(graph[..., np.newaxis])


def test_SBM_inputs():
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


def test_DCSBM_inputs():
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


def test_ER_fit():
    np.random.seed(8888)
    p = 0.5
    graph = er_np(1000, 0.5, directed=True, loops=False)
    ere = EREstimator(directed=True, loops=False)
    ere.fit(graph)
    p_hat = ere.p_
    assert p_hat - p < 0.001


def test_SBM_fit():
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


def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels
