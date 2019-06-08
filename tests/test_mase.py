import numpy as np
import pytest
from numpy import allclose, array_equal
from numpy.linalg import norm
from numpy.testing import assert_allclose

from graspy.cluster.gclust import GaussianCluster
from graspy.embed.mase import MultipleASE
from graspy.simulations.simulations import er_np, sbm
from graspy.utils.utils import is_symmetric, symmetrize


def make_train_undirected(n=[20, 20], m=10):
    """
    Make 4 class training dataset
    n = number of vertices
    m = number of graphs from each class
    """
    p1 = [[1, 0], [0, 1]]
    p2 = [[0, 1], [1, 0]]
    p3 = [[1, 0], [0, 0]]
    p4 = [[0, 0], [0, 1]]

    A = [sbm(n, p) for _ in range(m) for p in [p1, p2, p3, p4]]

    return A


def make_train_directed(n=[20, 20], m=10):
    p1 = [[0, 1], [0, 0]]
    p2 = [[0, 0], [1, 0]]
    p3 = [[1, 1], [0, 0]]
    p4 = [[0, 0], [1, 1]]

    A = [sbm(n, p, directed=True) for _ in range(m) for p in [p1, p2, p3, p4]]

    return A


def test_bad_inputs():
    with pytest.raises(TypeError):
        "Invalid unscaled"
        unscaled = "1"
        mase = MultipleASE(unscaled=unscaled)

    with pytest.raises(ValueError):
        "Test single graph input"
        np.random.seed(1)
        A = er_np(100, 0.2)
        MultipleASE().fit(A)

    with pytest.raises(ValueError):
        "Test graphs with different sizes"
        np.random.seed(1)
        A = [er_np(100, 0.2)] + [er_np(200, 0.2)]
        MultipleASE().fit(A)


def test_graph_clustering():
    """
    There should be 4 total clusters since 4 class problem.
    n_components = 2
    """
    # undirected case
    np.random.seed(2)
    n = [20, 20]
    m = 10
    X = make_train_undirected(n, m)

    res = MultipleASE(n_components=2).fit(X).scores_.reshape((m * 4, -1))
    gmm = GaussianCluster(10).fit(res)
    assert gmm.n_components_ == 4

    # directed case
    np.random.seed(3)
    X = make_train_directed(n, m)

    res = MultipleASE(n_components=2).fit(X).scores_.reshape((m * 4, -1))
    gmm = GaussianCluster(10).fit(res)
    assert gmm.n_components_ == 4

    # Scaled cases
    # undirected case
    np.random.seed(12)
    X = make_train_undirected(n, m)

    res = (
        MultipleASE(n_components=2, unscaled=False).fit(X).scores_.reshape((m * 4, -1))
    )
    gmm = GaussianCluster(10).fit(res)
    assert gmm.n_components_ == 4

    # directed case
    np.random.seed(13)
    X = make_train_directed(n, m)

    res = (
        MultipleASE(n_components=2, unscaled=False).fit(X).scores_.reshape((m * 4, -1))
    )
    gmm = GaussianCluster(10).fit(res)
    assert gmm.n_components_ == 4


def test_vertex():
    """
    There should be 2 clusters for undirected and 4 for directed.
    """
    # undirected case
    np.random.seed(4)
    n = [20, 20]
    m = 10
    X = make_train_undirected(n, m)

    res = MultipleASE(n_components=2).fit(X).latent_left_
    gmm = GaussianCluster(1, 10).fit(res)
    assert gmm.n_components_ == 2

    # directed case
    np.random.seed(5)
    X = make_train_directed(n, m)

    mase = MultipleASE(n_components=2).fit(X)
    res = np.hstack([mase.latent_left_, mase.latent_right_])
    gmm = GaussianCluster(1, 10).fit(res)
    assert gmm.n_components_ == 4

    # Scaled cases
    # undirected case
    np.random.seed(4)
    X = make_train_undirected(n, m)

    res = MultipleASE(n_components=2, unscaled=False).fit_transform(X)
    gmm = GaussianCluster(1, 10).fit(res)
    assert gmm.n_components_ == 2

    # directed case
    np.random.seed(5)
    X = make_train_directed(n, m)

    left, right = MultipleASE(n_components=2, unscaled=False).fit_transform(X)
    gmm = GaussianCluster(1, 10).fit(np.hstack([left, right]))
    assert gmm.n_components_ == 2  # why is this 2?
