import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from graspy.cluster.gclust import GaussianCluster


def test_inputs():
    # Generate random data
    X = np.random.normal(0, 1, size=(10, 2))

    # max_cluster > n_samples
    with pytest.raises(ValueError):
        gclust = GaussianCluster(100)
        gclust.fit(X)

    with pytest.raises(ValueError):
        gclust = GaussianCluster(100)
        gclust.fit_predict(X)

    # max_cluster < 0
    with pytest.raises(ValueError):
        gclust = GaussianCluster(max_components=-1)
        gclust.fit(X)

    with pytest.raises(ValueError):
        gclust = GaussianCluster(max_components=-1)
        gclust.fit_predict(X)


def test_outputs():
    np.random.seed(1)

    n = 100
    d = 3

    X1 = np.random.normal(2, 1, size=(n, d))
    X2 = np.random.normal(-2, 1, size=(n, d))
    X = np.vstack((X1, X2))
    y = np.repeat([0, 1], n)

    gclust = GaussianCluster(max_components=3)
    gclust.fit(X, y)
    bics = gclust.bic_
    aris = gclust.ari_

    assert_equal(np.argmin(bics), 1)
    assert_allclose(aris[1], 1)