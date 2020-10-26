# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from anytree import NodeMixin

from .autogmm import AutoGMMCluster
from .kclust import KMeansCluster


def _check_common_inputs(min_components, max_components, cluster_kws):
    if not isinstance(min_components, int):
        raise TypeError("min_components must be an int")
    elif min_components < 1:
        raise ValueError("min_components must be > 0")

    if not isinstance(max_components, int):
        raise TypeError("max_components must be an int")
    elif max_components < 1:
        raise ValueError("max_components must be > 0")
    elif max_components < min_components:
        raise ValueError("max_components must be >= min_components")

    if not isinstance(cluster_kws, dict):
        raise TypeError("cluster_kws must be a dict")


class DivisiveCluster(NodeMixin, BaseEstimator):
    """
    Recursively clusters data based on a chosen clustering algorithm.
    This algorithm implements a "divisive" or "top-down" approach.

    Parameters
    ----------
    cluster_method : str {"gmm", "kmeans"}, defaults to "gmm".
        The underlying clustering method to apply. If "gmm" will use
        :class:`~graspologic.cluster.AutoGMMCluster`. If "kmeans", will use
        :class:`~graspologic.cluster.KMeansCluster`.
    min_components : int, defaults to 1.
        The minimum number of mixture components/clusters to consider
        for the first split if "gmm" is selected as ``cluster_method``;
        and is set to 1 for later splits.
        If ``cluster_method`` is "kmeans", it is set to 2 for all splits.
    max_components : int, defaults to 2.
        The maximum number of mixture components/clusters to consider
        at each split.
    min_split : int, defaults to 1.
        The minimum size of a cluster for it to be considered to be split again.
    max_level : int, defaults to 4.
        The maximum number of times to recursively cluster the data.
    delta_criter : float, non-negative, defaults to 0.
        The smallest difference between selection criterion values of a new
        model and the current model that is required to accept the new model.
        Applicable only if ``cluster_method`` is "gmm".
    cluster_kws : dict, defaults to {}
        Keyword arguments (except ``min_components`` and ``max_components``) for chosen
        clustering method.

    Attributes
    ----------
    model_ : GaussianMixture or KMeans object
        Fitted clustering object based on which ``cluster_method`` was used.

    See Also
    --------
    graspologic.cluster.AutoGMMCluster
    graspologic.cluster.KMeansCluster
    anytree.node.nodemixin.NodeMixin

    Notes
    -----
    This class inherits from :class:`anytree.node.nodemixin.NodeMixin`, a lightweight
    class for doing various simple operations on trees.

    This algorithm was strongly inspired by maggotcluster, a divisive
    clustering algorithm in https://github.com/neurodata/maggot_models and the
    algorithm for estimating a hierarchical stochastic block model presented in [2]_.

    References
    ----------
    .. [1]  Athey, T. L., & Vogelstein, J. T. (2019).
            AutoGMM: Automatic Gaussian Mixture Modeling in Python.
            arXiv preprint arXiv:1909.02688.
    .. [2]  Lyzinski, V., Tang, M., Athreya, A., Park, Y., & Priebe, C. E
            (2016). Community detection and classification in hierarchical
            stochastic blockmodels. IEEE Transactions on Network Science and
            Engineering, 4(1), 13-26.
    """

    def __init__(
        self,
        cluster_method="gmm",
        min_components=1,
        max_components=2,
        cluster_kws={},
        min_split=1,
        max_level=4,
        delta_criter=0,
    ):

        _check_common_inputs(min_components, max_components, cluster_kws)

        if cluster_method not in ["gmm", "kmeans"]:
            msg = "clustering method must be one of"
            msg += "{gmm, kmeans}"
            raise ValueError(msg)

        if delta_criter < 0:
            raise ValueError("delta_criter must be non-negative")

        self.parent = None
        self.min_components = min_components
        self.max_components = max_components
        self.cluster_method = cluster_method
        self.cluster_kws = cluster_kws
        self.min_split = min_split
        self.max_level = max_level
        self.delta_criter = delta_criter

    def fit(self, X):
        """
        Fits clustering models to the data as well as resulting clusters

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.fit_predict(X)
        return self

    def fit_predict(self, X):
        """
        Fits clustering models to the data as well as resulting clusters
        and using fitted models to predict a hierarchy of labels

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        labels : array_label, shape (n_samples, n_levels)
        """
        X = check_array(X, dtype=[np.float64, np.float32], ensure_min_samples=1)

        if self.max_components > X.shape[0]:
            msg = "max_components must be >= n_samples, but max_components = "
            msg += "{}, n_samples = {}".format(self.max_components, X.shape[0])
            raise ValueError(msg)

        labels = self._fit(X)
        # delete the last column if predictions at the last level
        # are all zero vectors
        if (labels.shape[1] > 1) and (np.max(labels[:, -1]) == 0):
            labels = labels[:, :-1]

        return labels

    def _cluster_and_decide(self, X):
        if self.is_root:
            min_components = self.min_components
        else:
            min_components = 1

        if self.cluster_method == "gmm":
            cluster = AutoGMMCluster(
                min_components=min_components,
                max_components=self.max_components,
                **self.cluster_kws
            )
            cluster.fit(X)
            model = cluster.model_
            criter = cluster.criter_
            k = cluster.n_components_
            pred = cluster.predict(X)

            if self.delta_criter > 0:
                single_cluster = AutoGMMCluster(
                    min_components=1, max_components=1, **self.cluster_kws
                )
                single_cluster.fit(X)
                criter_single_cluster = single_cluster.criter_

                if k > 1:
                    # check whether the difference between the criterion
                    # of "split" and "not split" is greater than
                    # the threshold, delta_criter
                    if criter_single_cluster - criter < self.delta_criter:
                        pred = np.zeros((len(X), 1), dtype=int)

        elif self.cluster_method == "kmeans":
            cluster = KMeansCluster(
                max_clusters=self.max_components, **self.cluster_kws
            )
            cluster.fit(X)
            model = cluster.model_
            pred = cluster.predict(X)

        self.model_ = model
        return pred

    def _fit(self, X):
        pred = self._cluster_and_decide(X)
        self.children = []

        uni_labels = np.unique(pred)
        labels = pred.reshape((-1, 1)).copy()
        if len(uni_labels) > 1:
            for ul in uni_labels:
                inds = pred == ul
                new_X = X[inds]
                dc = DivisiveCluster(
                    cluster_method=self.cluster_method,
                    max_components=self.max_components,
                    min_split=self.min_split,
                    max_level=self.max_level,
                    cluster_kws=self.cluster_kws,
                    delta_criter=self.delta_criter,
                )
                dc.parent = self
                if (
                    len(new_X) > self.max_components
                    and len(new_X) >= self.min_split
                    and self.depth + 1 < self.max_level
                ):
                    child_labels = dc._fit(new_X)
                    while labels.shape[1] <= child_labels.shape[1]:
                        labels = np.column_stack(
                            (labels, np.zeros((len(X), 1), dtype=int))
                        )
                    labels[inds, 1 : child_labels.shape[1] + 1] = child_labels

        return labels

    def predict(self, X):
        """
        Predicts a hierarchy of labels based on fitted models

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        labels : array-like, shape (n_samples, n_levels)
        """

        check_is_fitted(self, ["model_"], all_or_any=all)
        X = check_array(X, dtype=[np.float64, np.float32], ensure_min_samples=1)

        labels = self._predict_labels(X)
        return labels

    def _predict_labels(self, X):
        if not self.is_leaf:
            pred_labels = np.zeros((len(X), self.height), dtype=int)
            current_pred_labels = self.model_.predict(X)
            pred_labels[:, 0] = current_pred_labels
            for label in np.unique(current_pred_labels):
                current_child = self.children[label]
                if not current_child.is_leaf:
                    child_pred_labels = current_child._predict_labels(
                        X[current_pred_labels == label]
                    )
                    pred_labels[
                        current_pred_labels == label, 1 : child_pred_labels.shape[1] + 1
                    ] = child_pred_labels
        else:
            # only for cases where root is a leaf cluster, i.e.,
            # only 1 cluster predicted at 1st level
            if self.is_root:
                pred_labels = np.zeros((len(X), 1), dtype=int)

        return pred_labels
