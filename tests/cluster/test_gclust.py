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
    assert_equal(np.argmin(bics), 1)


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

        # Assert that the two cluster model is the best
        assert_equal(np.argmin(bics), 1)
        assert_allclose(aris[np.argmin(bics)], 1)


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

        assert_equal(2, np.argmin(bics))
        assert_allclose(1, aris[np.argmin(bics)])
