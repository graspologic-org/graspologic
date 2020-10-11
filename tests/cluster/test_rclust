# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from sklearn.exceptions import NotFittedError
from sklearn.metrics import adjusted_rand_score

from graspologic.cluster.rclust import RecursiveCluster


def test_inputs():
    # Generate random data
    X = np.random.normal(0, 1, size=(10, 3))

    # min_components < 1
    with pytest.raises(ValueError):
        rc = RecursiveCluster(min_components=0)

    # min_components not integer
    with pytest.raises(TypeError):
        rc = RecursiveCluster(min_components="1")

    # max_components < min_components
    with pytest.raises(ValueError):
        rc = RecursiveCluster(min_components=1, max_components=0)

    # max_components not integer
    with pytest.raises(TypeError):
        rc = RecursiveCluster(max_components="1")

    # cluster_method not in ['AutoGMM', 'KMeans', 'Spherical-KMeans']
    with pytest.raises(ValueError):
        rc = RecursiveCluster(cluster_method="graspologic")

    # delta_criter not positive
    with pytest.raises(ValueError):
        rc = RecursiveCluster(delta_criter=0)

    # likelihood_ratio not in (0,1)
    with pytest.raises(ValueError):
        rc = RecursiveCluster(likelihood_ratio=0)

    # cluster_kws not a dict
    with pytest.raises(TypeError):
        rc = RecursiveCluster(cluster_kws=0)

    # level not an int
    with pytest.raises(TypeError):
        rc = RecursiveCluster(max_components=2)
        rc.fit_predict(X, level="1")

    with pytest.raises(TypeError):
        rc = RecursiveCluster(max_components=2)
        rc.fit(X)
        rc.predict(X, level="1")

    # level not positive
    with pytest.raises(ValueError):
        rc = RecursiveCluster(max_components=2)
        rc.fit_predict(X, level=0)

    with pytest.raises(TypeError):
        rc = RecursiveCluster(max_components=2)
        rc.fit(X)
        rc.predict(X, level=0)

    # max_components > n_sample
    with pytest.raises(ValueError):
        rc = RecursiveCluster(max_components=101)
        rc.fit(X)


def test_predict_without_fit():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    with pytest.raises(NotFittedError):
        rc = RecursiveCluster(max_components=2)
        rc.predict(X)


def test_two_class():
    """
    Easily separable two gaussian problem.
    """
    np.random.seed(1)

    n = 100
    d = 3

    X1 = np.random.normal(2, 0.5, size=(n, d))
    X2 = np.random.normal(-2, 0.5, size=(n, d))
    X = np.vstack((X1, X2))
    y = np.repeat([0, 1], n)

    rc = RecursiveCluster(max_components=5)
    rc.fit(X)

    # Assert that the two cluster model is the best at level 1
    assert_equal(rc.k_, 2)

    pred = rc.predictions[:, 0]
    ari = adjusted_rand_score(y, pred)
    # Assert that we get perfect clustering at level 1
    assert_allclose(ari, 1)
