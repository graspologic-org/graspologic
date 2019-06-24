import numpy as np
import pytest

from graspy.cluster.gclust import GaussianCluster
from graspy.embed.mase import MultipleASE
from graspy.simulations.simulations import er_np, sbm


def make_train_undirected(n=[128, 128], m=10, alpha=1):
    """
    Make 4 class training dataset
    n = number of vertices
    m = number of graphs from each class
    """
    c1 = np.array([[0.1, 0], [0, 0.1]])
    c2 = -1 * c1
    c3 = np.array([[0.1, 0], [0, 0]])
    c4 = np.array([[0, 0], [0, 0.1]])

    A = [
        sbm(n, np.ones((2, 2)) * 0.25 + alpha * c)
        for _ in range(m)
        for c in [c1, c2, c3, c4]
    ]

    return A


def make_train_directed(n=[128, 128], m=10):
    p1 = [[0, 0.9], [0, 0]]
    p2 = [[0, 0], [0.9, 0]]
    p3 = [[0.9, 0.9], [0, 0]]
    p4 = [[0, 0], [0.9, 0.9]]

    A = [sbm(n, p, directed=True) for _ in range(m) for p in [p1, p2, p3, p4]]

    return A


def test_bad_inputs():
    np.random.seed(1)
    single_graph = er_np(100, 0.2)
    different_size_graphs = [er_np(100, 0.2)] + [er_np(200, 0.2)]

    with pytest.raises(TypeError):
        "Invalid unscaled"
        mase = MultipleASE(scaled="1")

    with pytest.raises(ValueError):
        "Test single graph input"
        MultipleASE().fit(single_graph)

    with pytest.raises(ValueError):
        "Test 3-d tensor with 1 graph"
        single_graph_tensor = single_graph.reshape(1, 100, -1)
        MultipleASE().fit(single_graph_tensor)

    with pytest.raises(ValueError):
        "Empty list"
        MultipleASE().fit([])

    with pytest.raises(ValueError):
        "Test graphs with different sizes"
        MultipleASE().fit(different_size_graphs)


def test_graph_clustering():
    """
    There should be 4 total clusters since 4 class problem.
    n_components = 2
    """
    # undirected case
    np.random.seed(2)
    n = [128, 128]
    m = 10
    X = make_train_undirected(n, m)

    res = MultipleASE(2).fit(X).scores_.reshape((m * 4, -1))
    gmm = GaussianCluster(10, covariance_type="all").fit(res)
    assert gmm.n_components_ == 4

    # directed case
    np.random.seed(3)
    X = make_train_directed(n, m)

    res = MultipleASE(2).fit(X).scores_.reshape((m * 4, -1))
    gmm = GaussianCluster(10, covariance_type="all").fit(res)
    assert gmm.n_components_ == 4

    # Scaled cases
    # undirected case
    np.random.seed(12)
    X = make_train_undirected(n, m)

    res = MultipleASE(2, scaled=True).fit(X).scores_.reshape((m * 4, -1))
    gmm = GaussianCluster(10, covariance_type="all").fit(res)
    assert gmm.n_components_ == 4

    # directed case
    np.random.seed(13)
    X = make_train_directed(n, m)

    res = MultipleASE(2, scaled=True).fit(X).scores_.reshape((m * 4, -1))
    gmm = GaussianCluster(10, covariance_type="all").fit(res)
    assert gmm.n_components_ == 4


def test_vertex():
    """
    There should be 2 clusters since each graph is a 2 block model
    """
    # undirected case
    np.random.seed(4)
    n = [128, 128]
    m = 10
    X = make_train_undirected(n, m)

    # undirected case
    res = MultipleASE(n_components=2).fit(X).latent_left_
    gmm = GaussianCluster(10, covariance_type="all").fit(res)
    assert gmm.n_components_ == 2

    # directed case
    np.random.seed(5)
    X = make_train_directed(n, m)

    mase = MultipleASE(n_components=2).fit(X)
    res = np.hstack([mase.latent_left_, mase.latent_right_])
    gmm = GaussianCluster(10, covariance_type="all").fit(res)
    assert gmm.n_components_ == 2

    # Scaled and undirected case
    np.random.seed(4)
    X = make_train_undirected(n, m)

    res = MultipleASE(n_components=2, scaled=True).fit_transform(X)
    gmm = GaussianCluster(10, covariance_type="all").fit(res)
    assert gmm.n_components_ == 2

    # Scaled and directed case
    np.random.seed(5)
    X = make_train_directed(n, m)

    left, right = MultipleASE(n_components=2, scaled=True).fit_transform(X)
    res = np.hstack([left, right])
    gmm = GaussianCluster(10, covariance_type="all").fit(res)
    assert gmm.n_components_ == 2
