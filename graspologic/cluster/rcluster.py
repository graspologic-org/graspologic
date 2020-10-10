# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from anytree import NodeMixin

from .kclust import KMeansCluster
from .autogmm import AutoGMMCluster
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.base import BaseEstimator
from scipy.stats import chi2


def _check_common_inputs(min_components, max_components, cluster_kws, embed_kws):
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

    if embed_kws:
        if not isinstance(embed_kws, dict):
            raise TypeError("embed_kws must be a dict")

    if not isinstance(cluster_kws, dict):
        raise TypeError("cluster_kws must be a dict")


class RecursiveCluster(NodeMixin, BaseEstimator):
    """
    Recursively clusters based on a chosen clustering algorithm

    Parameters
    ----------
    cluster_method : str {"GMM", "KMeans", "Spherical-KMeans"}, default="GMM"
        The clustering method chosen to apply
    min_components : int, optional (default=1)
        The minimum number of mixture components or clusters to consider
        for the first split
    max_components : int, optional (default=10)
        The maximum number of mixture components or clusters to consider
        at each split
    selection_criteria : str, optional, default(=None) is the default
        selection_criteria of chosen clustering algorithm
        select the best model based on a certain criterion for each split
    min_split : int, optional (default=1)
        The minimum number of samples allowed in a leaf cluster
    max_level : int, optional (default=20)
        The maximum level to cluster
    delta_criter : float or None, positive, default=None
        The smallest difference between selection criterion values of a new
        model and the current model that is required to accept the new model
    likelihood_ratio : float or None, in range (0,1), default=None
        The significance threshold of p-value for likelihood ratio test

    Attributes
    ----------
    results_ : dict
        Contains information about clustering results on a cluster
        Items are:

        'model' : GaussianMixture (or KMeans) object if "GMM" (or "KMeans")
            is selected to perform clustering
        'criter' : float
            Bayesian (or Akaike) Information Criterion if "bic" (or "aic")
            is chosen to select the best model
        'n_components' : int
            number of components or clusters
    predictions : array_like, shape (n_samples, n_levels)

    See Also
    --------
    graspy.cluster.AutoGMMCluster

    References
    ----------
    .. [1]  Lyzinski, V., Tang, M., Athreya, A., Park, Y., & Priebe, C. E
            (2016). Community detection and classification in hierarchical
            stochastic blockmodels. IEEE Transactions on Network Science and
            Engineering, 4(1), 13-26.
    """

    def __init__(
        self,
        selection_criteria=None,
        cluster_method="GMM",
        parent=None,
        min_components=1,
        max_components=10,
        cluster_kws={},
        min_split=1,
        max_level=20,
        delta_criter=None,
        likelihood_ratio=None,
    ):

        _check_common_inputs(
            min_components, max_components, cluster_kws, embed_kws=None
        )

        if cluster_method not in ["GMM", "KMeans", "Spherical-KMeans"]:
            msg = "clustering method must be one of"
            msg += "{GMM, Kmeans, Spherical-KMeans}"
            raise ValueError(msg)

        if delta_criter:
            if delta_criter <= 0:
                raise ValueError("delta_criter must be positive")

        if likelihood_ratio:
            if likelihood_ratio <= 0 or likelihood_ratio >= 1:
                raise ValueError("likelihood_ratio must be in (0,1)")

        self.parent = parent
        self.min_components = min_components
        self.max_components = max_components
        self.cluster_method = cluster_method
        self.min_split = min_split
        self.selection_criteria = selection_criteria
        self.cluster_kws = cluster_kws
        self.min_split = min_split
        self.max_level = max_level
        self.delta_criter = delta_criter
        self.likelihood_ratio = likelihood_ratio

    def fit(self, X):
        """
        Fits clustering models to the data as well as resulting clusters

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.fit_predict(X)
        return self

    def fit_predict(self, X, level=None):
        """
        Fits clustering models to the data as well as resulting clusters
        and using fitted models to predict a hierarchy of labels

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        level: int, optional (default=None)

        Returns
        -------
        labels : array_label, shape (n_samples, n_levels)
            if no level specified; otherwise, shape (n_samples,)
        """
        X = check_array(X, dtype=[np.float64, np.float32], ensure_min_samples=1)
        if level:
            if not isinstance(level, int) or level < 1:
                raise TypeError("level must be a positive int")

        self.X = X
        self.labels = np.zeros((len(X), 1), dtype=int)
        self.indx = np.arange(len(X))

        if self.max_components > X.shape[0]:
            msg = "max_components must be >= n_samples, but max_components = "
            msg += "{}, n_samples = {}".format(self.max_components, X.shape[0])
            raise ValueError(msg)

        while True:
            current_node = self._get_next_node()
            if current_node:
                current_node._cluster_and_decide()
            else:
                break

        labels = self._unique_labels(self.labels)
        if level:
            if level <= labels.shape[1]:
                labels = labels[:, level - 1]
            else:
                msg = "input exceeds max level = {}".format(labels.shape[1])
                raise ValueError(msg)

        self.predictions = labels

        return labels

    def _get_next_node(self):
        leaves = self.root.leaves
        current_node = []

        for leaf in leaves:
            if (
                len(leaf.X) >= self.max_components
                and len(leaf.X) >= self.min_split
                and leaf.depth < self.max_level
            ):
                if not hasattr(leaf, "k_"):
                    current_node = leaf
                    break
        return current_node

    def _cluster_and_decide(self):
        X = self.X
        if self.is_root:
            min_components = self.min_components
        else:
            min_components = 1

        if self.cluster_method == "GMM":
            cluster = AutoGMMCluster(
                min_components=min_components,
                max_components=self.max_components,
                **self.cluster_kws
            )
            cluster.fit(X)
            model = cluster.model_
            criter = cluster.criter_
            lik = model.score(X)
            k = cluster.n_components_
            pred = cluster.predict(X)

            if self.delta_criter or self.likelihood_ratio:
                single_cluster = AutoGMMCluster(
                    min_components=1, max_components=1, **self.cluster_kws
                )
                single_cluster.fit(X)
                criter_single_cluster = single_cluster.criter_
                lik_single_cluster = single_cluster.model_.score(X)

            if k > 1:
                # check whether the difference between the criterion of "split"
                # and "not split" is greater than the threshold, delta_criter
                if self.delta_criter:
                    if criter_single_cluster - criter < self.delta_criter:
                        k = 1
                # perform likelihood ratio test
                if self.likelihood_ratio:
                    LR = 2 * (lik - lik_single_cluster)
                    p = chi2.sf(LR, k - 1)
                    # TODO: maybe set a default p-value threshold if do LR test
                    if p < self.likelihood_ratio:
                        k = 1
                # TODO: maybe add similar tests for Kmeans

        elif self.cluster_method == "Kmeans":
            cluster = KMeansCluster(
                max_clusters=self.max_components, **self.cluster_kws
            )
            cluster.fit(X)
            model = cluster.model_
            k = cluster.n_clusters_
            criter = cluster.silhouette_
            pred = cluster.predict(X)

        elif self.cluster_method == "Spherical-Kmeans":
            # TODO: could use SphericalKMeans
            pass

        results = {"model": model, "criter": criter, "n_components": k}
        self.results_ = results
        self.k_ = k
        self.model_ = model
        self.pred_ = pred
        self.children = []

        if k > 1:
            if self.root.labels.shape[1] < self.depth + 1:
                self.root.labels = np.column_stack(
                    (self.root.labels, np.zeros((len(self.root.X), 1), dtype=int))
                )
            self.root.labels[self.indx, self.depth] = pred

            uni_labels = np.unique(pred)
            for ul in uni_labels:
                inds = pred == ul
                new_x = self.X[inds]

                RecursiveCluster(
                    selection_criteria=self.selection_criteria,
                    cluster_method=self.cluster_method,
                    parent=self,
                    max_components=self.max_components,
                    min_split=self.min_split,
                    max_level=self.max_level,
                    cluster_kws=self.cluster_kws,
                    delta_criter=self.delta_criter,
                    likelihood_ratio=self.likelihood_ratio,
                )
                self.children[-1].X = new_x
                self.children[-1].indx = self.indx[
                    inds
                ]  # the index wrt original data samples

    def predict(self, X, level=None):
        """
        Predicts a hierarchy of labels based on fitted models

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
        level: int, optional (default=None)
            the level of flat clustering to generate

        Returns
        -------
        labels : array_like, shape (n_samples, n_levels)
            if no level specified; otherwise, shape (n_samples,)
        """

        check_is_fitted(self, ["model_"], all_or_any=all)

        X = check_array(X, dtype=[np.float64, np.float32], ensure_min_samples=1)
        if level is not None:
            if not isinstance(level, int) or level < 1:
                raise TypeError("level must be a positive int")

        labels = self._predict_labels(X)
        labels = self._unique_labels(labels)

        if level:
            if level <= labels.shape[1]:
                labels = labels[:, level - 1]
            else:
                msg = "input exceeds max level = {}".format(labels.shape[1])
                raise ValueError(msg)

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

    def _unique_labels(self, labels):
        # re-number labeling so that labels[:,i] assigns a unique value to each cluster
        n_levels = labels.shape[1]
        for level in range(0, n_levels):
            # to avoid updated labels accidentally matching old ones
            new_labels = -labels[:, : level + 1] - 1
            uni_labels = np.unique(new_labels, axis=0)
            for ul in range(len(uni_labels)):
                uni_labels_inds = np.where(
                    (uni_labels[:, None, :] == new_labels).all(2)[ul]
                )[0]
                labels[uni_labels_inds, level] = ul

        return labels
