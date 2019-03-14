import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from sklearn.exceptions import NotFittedError

from graspy.cluster.gclust import GaussianCluster
from graspy.embed.ase import AdjacencySpectralEmbed
from graspy.simulations.simulations import sbm


def test_inputs():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    with pytest.raises(TypeError):
        gclust = GaussianCluster(max_components="1")

    # max_cluster > n_samples
    with pytest.raises(ValueError):
        gclust = GaussianCluster(1000)
        gclust.fit(X)

    with pytest.raises(ValueError):
        gclust = GaussianCluster(1000)
        gclust.fit_predict(X)

    # max_cluster < 0
    with pytest.raises(ValueError):
        gclust = GaussianCluster(max_components=-1)
        gclust.fit(X)

    with pytest.raises(ValueError):
        gclust = GaussianCluster(max_components=-1)
        gclust.fit_predict(X)

    with pytest.raises(ValueError):
        gclust = GaussianCluster(min_components=-1)
        gclust.fit(X)

    with pytest.raises(ValueError):
        gclust = GaussianCluster(max_components=-1)
        gclust.fit_predict(X)

    with pytest.raises(ValueError):
        gclust = GaussianCluster(min_components=2, max_components=1)
        gclust.fit(X)

    with pytest.raises(ValueError):
        gclust = GaussianCluster(min_components=2, max_components=1)
        gclust.fit_predict(X)


def test_predict_without_fit():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    with pytest.raises(NotFittedError):
        gclust = GaussianCluster(max_components=2)
        gclust.predict(X)


def test_no_y():
    np.random.seed(2)

    n = 100
    d = 3

    X1 = np.random.normal(2, 0.5, size=(n, d))
    X2 = np.random.normal(-2, 0.5, size=(n, d))
    X = np.vstack((X1, X2))

    gclust = GaussianCluster(max_components=5)
    gclust.fit(X)

    bics = gclust.bic_
    assert_equal(bics.iloc[:, 0].values.argmin(), 1)


def test_outputs():
    """
    Easily separable two gaussian problem.
    """
    np.random.seed(2)

    n = 100
    d = 3

    num_sims = 10

    for _ in range(num_sims):
        X1 = np.random.normal(2, 0.5, size=(n, d))
        X2 = np.random.normal(-2, 0.5, size=(n, d))
        X = np.vstack((X1, X2))
        y = np.repeat([0, 1], n)

        gclust = GaussianCluster(max_components=5)
        gclust.fit(X, y)
        bics = gclust.bic_
        aris = gclust.ari_

        bic_argmin = bics.iloc[:, 0].values.argmin()

        # Assert that the two cluster model is the best
        assert_equal(bic_argmin, 1)
        # The plus one is to adjust the index by min_components
        assert_allclose(aris.iloc[:, 0][bic_argmin + 1], 1)


def test_bic():
    """
    Expect 3 clusters from a 3 block model
    """
    np.random.seed(3)
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
        gclust = GaussianCluster(max_components=10)
        gclust.fit(X_hat, y)

        bics = gclust.bic_
        aris = gclust.ari_

        bic_argmin = bics.iloc[:, 0].values.argmin()

        assert_equal(2, bic_argmin)
        # The plus one is to adjust the index by min_components
        assert_allclose(1, aris.iloc[:, 0][bic_argmin + 1])


def test_covariances():
    """
    Easily separable two gaussian problem.
    """
    np.random.seed(2)

    n = 100

    mu1 = [-10, 0]
    mu2 = [10, 0]

    # Spherical
    cov1 = 2 * np.eye(2)
    cov2 = 2 * np.eye(2)

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    gclust_object = GaussianCluster(max_components=2, covariance_type="all")
    gclust_object.fit(X)
    assert_equal(gclust_object.bic_.iloc[1, :].values.argmin(), 0)

    # Diagonal
    np.random.seed(10)
    cov1 = np.diag([1, 1])
    cov2 = np.diag([2, 1])

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    gclust_object = GaussianCluster(max_components=2, covariance_type="all")
    gclust_object.fit(X)
    assert_equal(gclust_object.bic_.iloc[1, :].values.argmin(), 1)

    # Tied
    cov1 = np.array([[2, 1], [1, 2]])
    cov2 = np.array([[2, 1], [1, 2]])

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    gclust_object = GaussianCluster(max_components=2, covariance_type="all")
    gclust_object.fit(X)
    assert_equal(gclust_object.bic_.iloc[1, :].values.argmin(), 2)

    # Full
    cov1 = np.array([[2, -1], [-1, 2]])
    cov2 = np.array([[2, 1], [1, 2]])

    X1 = np.random.multivariate_normal(mu1, cov1, n)
    X2 = np.random.multivariate_normal(mu2, cov2, n)

    X = np.concatenate((X1, X2))

    gclust_object = GaussianCluster(max_components=2, covariance_type="all")
    gclust_object.fit(X)
    assert_equal(gclust_object.bic_.iloc[1, :].values.argmin(), 3)
