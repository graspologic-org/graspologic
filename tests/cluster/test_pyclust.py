import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from sklearn.exceptions import NotFittedError

from graspy.cluster.pyclust import PyclustCluster
from graspy.embed.ase import AdjacencySpectralEmbed
from graspy.simulations.simulations import sbm


def test_inputs():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    # min_components < 1
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(min_components=0)

    # min_components integer
    with pytest.raises(TypeError):
        pyclust = PyclustCluster(min_components="1")

    # max_components < min_components
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(min_components=1, max_components=0)

    # max_components integer
    with pytest.raises(TypeError):
        pyclust = PyclustCluster(min_components=1, max_components="1")

    # affinity is not an array, string or list
    with pytest.raises(TypeError):
        pyclust = PyclustCluster(min_components=1, affinity=1)

    # affinity is not in ['euclidean', 'manhattan', 'cosine', 'none']
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(min_components=1, affinity="graspy")

    # linkage is not an array, string or list
    with pytest.raises(TypeError):
        pyclust = PyclustCluster(min_components=1, linkage=1)

    # linkage is not in ['single', 'average', 'complete', 'ward']
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(min_components=1, linkage="graspy")

    # covariance type is not an array, string or list
    with pytest.raises(TypeError):
        pyclust = PyclustCluster(min_components=1, covariance_type=1)

    # covariance type is not in ['spherical', 'diag', 'tied', 'full']
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(min_components=1, covariance_type="graspy")

    # min_cluster > n_samples when max_cluster is None
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(1000)
        pyclust.fit(X)

    with pytest.raises(ValueError):
        pyclust = PyclustCluster(1000)
        pyclust.fit_predict(X)

    # max_cluster > n_samples when max_cluster is not None
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(10, 1001)
        pyclust.fit(X)

    with pytest.raises(ValueError):
        pyclust = PyclustCluster(10, 1001)
        pyclust.fit_predict(X)

    # min_cluster > n_samples when max_cluster is None
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(1000)
        pyclust.fit(X)

    with pytest.raises(ValueError):
        pyclust = PyclustCluster(10, 1001)
        pyclust.fit_predict(X)

    # min_cluster > n_samples when max_cluster is not None
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(1000, 1001)
        pyclust.fit(X)

    with pytest.raises(ValueError):
        pyclust = PyclustCluster(1000, 1001)
        pyclust.fit_predict(X)

    # label_init is not a 1-D array
    with pytest.raises(TypeError):
        pyclust = PyclustCluster(label_init=np.zeros([100, 2]))

    # label_init is not 1-D array, a list or None.
    with pytest.raises(TypeError):
        pyclust = PyclustCluster(label_init="label")

    # label_init length is not equal to n_samples
    with pytest.raises(ValueError):
        pyclust = PyclustCluster(label_init=np.zeros([50, 1]))
        pyclust.fit(X)

    with pytest.raises(ValueError):
        pyclust = PyclustCluster(label_init=np.zeros([50, 1]))
        pyclust.fit_predict(X)

    with pytest.raises(TypeError):
        pyclust = PyclustCluster(label_init=np.zeros([100, 2]), max_iter=-2)


def test_predict_without_fit():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    with pytest.raises(NotFittedError):
        pyclust = PyclustCluster(min_components=2)
        pyclust.predict(X)


def test_cosine_on_0():
    X = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 0, 1]])

    with pytest.raises(ValueError):
        pyclust = PyclustCluster(min_components=3, affinity="all")
        pyclust.fit(X)


def test_no_y():
    np.random.seed(1)

    n = 100
    d = 3

    X1 = np.random.normal(2, 0.5, size=(n, d))
    X2 = np.random.normal(-2, 0.5, size=(n, d))
    X = np.vstack((X1, X2))

    pyclust = PyclustCluster(min_components=5)
    pyclust.fit(X)

    assert_equal(pyclust.n_components_, 2)


def test_two_class():
    """
    Easily separable two gaussian problem.
    """
    np.random.seed(1)

    n = 100
    d = 3

    num_sims = 10

    for _ in range(num_sims):
        X1 = np.random.normal(2, 0.5, size=(n, d))
        X2 = np.random.normal(-2, 0.5, size=(n, d))
        X = np.vstack((X1, X2))
        y = np.repeat([0, 1], n)

        pyclust = PyclustCluster(min_components=5)
        pyclust.fit(X, y)

        n_components = pyclust.n_components_

        # Assert that the two cluster model is the best
        assert_equal(n_components, 2)

        # Asser that we get perfect clustering
        assert_allclose(pyclust.ari_, 1)


def test_five_class():
    """
    Easily separable five gaussian problem.
    """
    np.random.seed(1)

    n = 100
    mus = [[i * 5, 0] for i in range(5)]
    cov = np.eye(2)  # balls

    num_sims = 10

    for _ in range(num_sims):
        X = np.vstack([np.random.multivariate_normal(mu, cov, n) for mu in mus])

        pyclust = PyclustCluster(
            min_components=3, max_components=10, covariance_type="all"
        )
        pyclust.fit(X)

        assert_equal(pyclust.n_components_, 5)


def test_ase_three_blocks():
    """
    Expect 3 clusters from a 3 block model
    """
    np.random.seed(1)
    num_sims = 10

    # Generate adjacency and labels
    n = 50
    n_communites = [n, n, n]
    p = np.array([[0.8, 0.3, 0.2], [0.3, 0.8, 0.3], [0.2, 0.3, 0.8]])
    y = np.repeat([1, 2, 3], repeats=n)

    for _ in range(num_sims):
        A = sbm(n=n_communites, p=p)

        # Embed to get latent positions
        ase = AdjacencySpectralEmbed(n_components=5)
        X_hat = ase.fit_transform(A)

        # Compute clusters
        pyclust = PyclustCluster(min_components=10)
        pyclust.fit(X_hat, y)

        n_components = pyclust.n_components_

        # Assert that the three cluster model is the best
        assert_equal(n_components, 3)

        # Asser that we get perfect clustering
        assert_allclose(pyclust.ari_, 1)


def test_covariances():
    """
    Easily separable two gaussian problem.
    """
    np.random.seed(1)

    n = 100

    mu1 = [-10, 0]
    mu2 = [10, 0]

    # Spherical
    cov1 = 2 * np.eye(2)
    cov2 = 2 * np.eye(2)

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    pyclust = PyclustCluster(min_components=2, covariance_type="all")
    pyclust.fit(X)
    assert_equal(pyclust.covariance_type_, "spherical")

    # Diagonal
    np.random.seed(10)
    cov1 = np.diag([1, 1])
    cov2 = np.diag([2, 1])

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    pyclust = PyclustCluster(min_components=2, covariance_type="all")
    pyclust.fit(X)
    assert_equal(pyclust.covariance_type_, "diag")

    # Tied
    cov1 = np.array([[2, 1], [1, 2]])
    cov2 = np.array([[2, 1], [1, 2]])

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    pyclust = PyclustCluster(min_components=2, covariance_type="all")
    pyclust.fit(X)
    assert_equal(pyclust.covariance_type_, "tied")

    # Full
    cov1 = np.array([[2, -1], [-1, 2]])
    cov2 = np.array([[2, 1], [1, 2]])

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    pyclust = PyclustCluster(min_components=2, covariance_type="all")
    pyclust.fit(X)
    assert_equal(pyclust.covariance_type_, "full")
