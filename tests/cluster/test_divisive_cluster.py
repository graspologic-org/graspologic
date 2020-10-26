# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less, assert_equal
from sklearn.exceptions import NotFittedError
from sklearn.metrics import adjusted_rand_score

from graspologic.cluster import DivisiveCluster


def test_inputs():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    # min_components < 1
    with pytest.raises(ValueError):
        dc = DivisiveCluster(min_components=0)

    # min_components not integer
    with pytest.raises(TypeError):
        dc = DivisiveCluster(min_components="1")

    # max_components < min_components
    with pytest.raises(ValueError):
        dc = DivisiveCluster(min_components=1, max_components=0)

    # max_components not integer
    with pytest.raises(TypeError):
        dc = DivisiveCluster(max_components="1")

    # cluster_method not in ['gmm', 'kmeans']
    with pytest.raises(ValueError):
        dc = DivisiveCluster(cluster_method="graspologic")

    # delta_criter negative
    with pytest.raises(ValueError):
        dc = DivisiveCluster(delta_criter=-1)

    # cluster_kws not a dict
    with pytest.raises(TypeError):
        dc = DivisiveCluster(cluster_kws=0)

    # max_components > n_sample
    with pytest.raises(ValueError):
        dc = DivisiveCluster(max_components=101)
        dc.fit(X)


def test_predict_without_fit():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    with pytest.raises(NotFittedError):
        dc = DivisiveCluster(max_components=2)
        dc.predict(X)


def test_predict_on_nonfitted_data_gmm():
    # Generate random data to fit on
    np.random.seed(1)
    n = 100
    d = 3
    X1 = np.random.normal(1, 0.1, size=(n, d))
    X2 = np.random.normal(2, 0.1, size=(n, d))
    X = np.vstack((X1, X2))
    y = np.repeat([0, 1], n)

    dc = DivisiveCluster(max_components=2)
    pred1 = dc.fit_predict(X)

    # Generate random data to predict on
    np.random.seed(2)
    n = 50
    d = 3
    X1_new = np.random.normal(1, 0.1, size=(n, d))
    X2_new = np.random.normal(2, 0.1, size=(n, d))
    X_new = np.vstack((X1_new, X2_new))
    y_new = np.repeat([0, 1], n)

    pred2 = dc.predict(X_new)

    # Assert that both predictions have the same depth
    assert_equal(pred1.shape[1], pred2.shape[1])

    # Assert that both predictions represent a perfect clustering
    # of 2 clusters
    assert_equal(np.max(pred1) + 1, 2)
    ari_1 = adjusted_rand_score(y, pred1[:, 0])
    assert_allclose(ari_1, 1)

    assert_equal(np.max(pred2) + 1, 2)
    ari_2 = adjusted_rand_score(y_new, pred2[:, 0])
    assert_allclose(ari_2, 1)


def test_predict_on_nonfitted_data_kmeans():
    # Generate random data to fit on
    np.random.seed(1)
    n = 100
    d = 3
    X1 = np.random.normal(1, 0.1, size=(n, d))
    X2 = np.random.normal(2, 0.1, size=(n, d))
    X = np.vstack((X1, X2))
    y = np.repeat([0, 1], n)

    dc = DivisiveCluster(max_components=2, cluster_method="kmeans")
    pred1 = dc.fit_predict(X)

    # Generate random data to predict on
    np.random.seed(2)
    n = 50
    d = 3
    X1_new = np.random.normal(1, 0.1, size=(n, d))
    X2_new = np.random.normal(2, 0.1, size=(n, d))
    X_new = np.vstack((X1_new, X2_new))
    y_new = np.repeat([0, 1], n)

    pred2 = dc.predict(X_new)

    # Assert that both predictions have the same depth
    assert_equal(pred1.shape[1], pred2.shape[1])

    # Assert that both predictions represent a perfect clustering
    # of 2 clusters at 1st level
    assert_equal(np.max(pred1[:, 0]) + 1, 2)
    ari_1 = adjusted_rand_score(y, pred1[:, 0])
    assert_allclose(ari_1, 1)

    assert_equal(np.max(pred2[:, 0]) + 1, 2)
    ari_2 = adjusted_rand_score(y_new, pred2[:, 0])
    assert_allclose(ari_2, 1)

    # Assert that predictions on new data have the same or fewer
    # clusters than those on the fitted data at each level
    for lvl in range(pred1.shape[1]):
        n_cluster1 = np.max(pred1[:, lvl]) + 1
        n_cluster2 = np.max(pred2[:, lvl]) + 1
        assert_array_less(n_cluster2, n_cluster1 + 1)


#  Easily separable hierarchical data with 2 levels
#  of four gaussians
np.random.seed(1)
n = 100
d = 3

X11 = np.random.normal(-5, 0.1, size=(n, d))
X21 = np.random.normal(-2, 0.1, size=(n, d))
X12 = np.random.normal(2, 0.1, size=(n, d))
X22 = np.random.normal(5, 0.1, size=(n, d))
X = np.vstack((X11, X21, X12, X22))

# true labels of 2 levels
y_lvl1 = np.repeat([0, 1], 2 * n)
y_lvl2 = np.repeat([0, 1, 2, 3], n)


def _test_hierarchical_four_class(**kws):
    """
    Clustering above hierarchical data with gmm
    """
    np.random.seed(1)
    dc = DivisiveCluster(max_components=2, **kws)
    pred = dc.fit_predict(X)

    # re-number "pred" so that each column represents
    # a flat clustering at current level
    for lvl in range(1, pred.shape[1]):
        _, inds = np.unique(pred[:, : lvl + 1], axis=0, return_inverse=True)
        pred[:, lvl] = inds

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


def test_hierarchical_four_class_gmm():
    _test_hierarchical_four_class(cluster_method="gmm")


def test_hierarchical_four_class_aic():
    _test_hierarchical_four_class(cluster_kws=dict(selection_criteria="aic"))


def test_hierarchical_four_class_kmeans():
    _test_hierarchical_four_class(cluster_method="kmeans")


def test_hierarchical_six_class_delta_criter():
    """
    Clustering on less easily separable hierarchical data with 2 levels
    of six gaussians
    """

    np.random.seed(1)

    n = 100
    d = 3

    X11 = np.random.normal(-4, 0.8, size=(n, d))
    X21 = np.random.normal(-3, 0.8, size=(n, d))
    X31 = np.random.normal(-2, 0.8, size=(n, d))
    X12 = np.random.normal(2, 0.8, size=(n, d))
    X22 = np.random.normal(3, 0.8, size=(n, d))
    X32 = np.random.normal(4, 0.8, size=(n, d))
    X = np.vstack((X11, X21, X31, X12, X22, X32))

    y_lvl1 = np.repeat([0, 1], 3 * n)
    y_lvl2 = np.repeat([0, 1, 2, 3, 4, 5], n)

    # Perform clustering without setting delta_criter
    dc = DivisiveCluster(max_components=2)
    pred = dc.fit_predict(X)

    # re-number "pred" so that each column represents
    # a flat clustering at current level
    for lvl in range(1, pred.shape[1]):
        _, inds = np.unique(pred[:, : lvl + 1], axis=0, return_inverse=True)
        pred[:, lvl] = inds

    # Perform clustering while setting delta_criter
    dc = DivisiveCluster(max_components=2, delta_criter=10)
    pred_delta_criter = dc.fit_predict(X)

    # re-number "pred_delta_criter" so that each column represents
    # a flat clustering at current level
    for lvl in range(1, pred_delta_criter.shape[1]):
        _, inds = np.unique(
            pred_delta_criter[:, : lvl + 1], axis=0, return_inverse=True
        )
        pred_delta_criter[:, lvl] = inds

    # Assert that pred has more levels than pred_delta_criter
    assert_equal(pred.shape[1] - 1, pred_delta_criter.shape[1])

    # Assert that both pred_delta_criter and pred represent
    # perfect clustering at the first level
    ari_lvl1 = adjusted_rand_score(y_lvl1, pred[:, 0])
    assert_allclose(ari_lvl1, 1)
    ari_delta_criter_lvl1 = adjusted_rand_score(y_lvl1, pred_delta_criter[:, 0])
    assert_allclose(ari_delta_criter_lvl1, 1)

    # Assert that pred_delta_criter leads to a clustering as good as
    # pred at the second level
    ari_lvl2 = adjusted_rand_score(y_lvl2, pred[:, 1])
    ari_delta_criter_lvl2 = adjusted_rand_score(y_lvl2, pred_delta_criter[:, 1])
    assert_allclose(ari_delta_criter_lvl2, ari_lvl2)

    # Assert that pred suggests oversplitting at the last level (level 3)
    # which leads to a worse clustering than the last level
    # of pred_delta_criter (level 2)
    ari_lvl3 = adjusted_rand_score(y_lvl2, pred[:, -1])
    assert_array_less(ari_lvl3, ari_delta_criter_lvl2)
