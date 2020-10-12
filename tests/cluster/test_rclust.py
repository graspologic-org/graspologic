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


def test_hierarchical_four_class():
    """
    Easily separable hierarchical four-gaussian problem.
    """
    np.random.seed(1)

    n = 100
    d = 3

    X11 = np.random.normal(-5, 0.1, size=(n, d))
    X21 = np.random.normal(-2, 0.1, size=(n, d))
    X12 = np.random.normal(2, 0.1, size=(n, d))
    X22 = np.random.normal(5, 0.1, size=(n, d))
    X = np.vstack((X11, X21, X12, X22))
    y_lvl1 = np.repeat([0, 1], 2 * n)
    y_lvl2 = np.repeat([0, 1, 2, 3], n)

    rc = RecursiveCluster(max_components=2)
    pred = rc.fit_predict(X)

    # Assert that the 2-cluster model is the best at level 1
    assert_equal(np.max(pred[:, 0]) + 1, 2)
    # Assert that the 4-cluster model is the best at level 1
    assert_equal(np.max(pred[:, 1]) + 1, 4)

    # Assert that we get perfect clustering at level 1
    ari_lvl1 = adjusted_rand_score(y_lvl1, pred[:, 0])
    assert_allclose(ari_lvl1, 1)

    # Assert that we get perfect clustering at level 2
    ari_lvl2 = adjusted_rand_score(y_lvl2, pred[:, 1])
    assert_allclose(ari_lvl2, 1)
