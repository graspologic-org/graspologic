import numpy as np
from anytree import NodeMixin, LevelOrderIter
from anytree.search import findall

from graspy.cluster import KMeansCluster, AutoGMMCluster
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.base import BaseEstimator
from scipy.stats import chi2

# from spherecluster import SphericalKMeans


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
    Recursively cluster using a chosen clustering algorithm

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
        if level:
            if not isinstance(level, int):
                raise TypeError("level must be an int")

        # if already predited on some new data,
        # clean up attributes attched to corresponding nodes in the fitted tree
        for n in [node for node in LevelOrderIter(self)]:
            if hasattr(n, "new_x"):
                del n.new_x, n.new_inds, n.new_pred, n.predicted

        # the fitted node corresponding to the nxt node to predict
        nxt_fitted_node = [self]
        self.new_x = X
        self.new_inds = np.arange(len(X))

        while nxt_fitted_node:
            nxt_fitted_node = nxt_fitted_node[0]
            if hasattr(nxt_fitted_node, "model_") and not nxt_fitted_node.is_leaf:
                model = nxt_fitted_node.model_
                pred = model.predict(nxt_fitted_node.new_x)

                uni_labels = np.unique(pred)
                for ul in uni_labels:
                    inds = pred == ul
                    new_x = nxt_fitted_node.new_x[inds]
                    nxt_node = nxt_fitted_node.children[ul]
                    nxt_node.new_x = new_x
                    nxt_node.new_inds = inds

            # consider leaf nodes and nodes w/o "model_"
            else:
                # nodes w/o "model_" were not fitted b/c had too few samples
                # or were too deep so assume a model of single cluster here
                pred = np.zeros(len(nxt_fitted_node.new_x))

            # find available node to predict
            nxt_fitted_node.new_pred = pred
            nxt_fitted_node.predicted = True
            nodes_with_new_x = findall(self, lambda node: hasattr(node, "new_x"))
            nxt_fitted_node = [
                j for i, j in enumerate(nodes_with_new_x) if not hasattr(j, "predicted")
            ]

        labels = self._to_labels(X, True)
        if level:
            if level < labels.shape[1]:
                labels = labels[:, level]
            else:
                msg = "input exceeds max level = {}".format(labels.shape[1] - 1)
                raise ValueError(msg)

        return labels

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
            if not isinstance(level, int):
                raise TypeError("level must be an int")

        self.X = X

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

        labels = self._to_labels(self.X, False)
        if level:
            if level < labels.shape[1]:
                labels = labels[:, level]
            else:
                msg = "input exceeds max level = {}".format(labels.shape[1] - 1)
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

        results = {
            "model": model,
            # "pred": pred,
            "criter": criter,
            "n_components": k,
        }
        self.results_ = results
        self.k_ = k
        self.model_ = model
        self.pred_ = pred
        self.children = []

        if k > 1:
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
                self.children[-1].inds = inds

    def _to_labels(self, X, pred_on_new):
        n_levels = self.height
        n_sample = len(X)
        labels = np.zeros((n_sample, n_levels))

        children = [node for node in LevelOrderIter(self)][1:]
        pred_attr = "pred_"
        inds_attr = "inds"
        # if labels are calculated after predicting on new data
        if pred_on_new:
            children = [node for node in children if hasattr(node, pred_attr)]
            pred_attr = "new_pred"
            inds_attr = "new_inds"

        labels[:, 0] = getattr(self, pred_attr)
        # place predictions on nodes at appropriate locations in labels
        for c in range(len(children)):
            level = children[c].depth
            if level < n_levels and hasattr(children[c], pred_attr):
                path = children[c].path[1:]
                X_inds = np.arange(n_sample)
                for node in range(len(path)):
                    X_inds = X_inds[getattr(path[node], inds_attr)]
                labels[X_inds, level] = getattr(children[c], pred_attr)

        # renumber labeling so labels[:,i] assigns a uniq value to each cluster
        for level in range(1, n_levels):
            # to avoid updated labels accidentally matching old ones
            new_labels = -labels[:, : level + 1] - 1
            uni_labels = np.unique(new_labels, axis=0)
            for ul in range(len(uni_labels)):
                uni_labels_inds = np.where(
                    (uni_labels[:, None, :] == new_labels).all(2)[ul]
                )[0]
                labels[uni_labels_inds, level] = ul

        if pred_on_new:
            for lvl in range(labels.shape[1]):
                _, labels[:, lvl] = np.unique(labels[:, lvl], return_inverse=True)

        return labels
