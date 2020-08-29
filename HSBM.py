import numpy as np
import warnings

from anytree import NodeMixin, LevelOrderGroupIter, LevelOrderIter
# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.cluster import AgglomerativeClustering

from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed, select_dimension
from graspy.cluster import KMeansCluster, AutoGMMCluster
# from graspy.cluster.base import BaseCluster
# from graspy.models import SBMEstimator
from graspy.models.sbm import _calculate_block_p, _block_to_full, _get_block_indices
from graspy.models.base import BaseGraphEstimator
from graspy.utils import augment_diagonal, pass_to_ranks, import_graph, remove_loops, symmetrize

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y
from sklearn.base import BaseEstimator
from scipy.stats import chi2
# from spherecluster import SphericalKMeans


def _check_common_inputs(
    min_components, max_components, cluster_kws, embed_kws
):
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
    def __init__(
        self,
        selection_criteria=None,
        cluster_method="GMM",
        n_init=1,
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
            msg = "clustering method must be one of {GMM, Kmeans, Spherical-KMeans}"
            raise ValueError(msg)

        if delta_criter:
            if delta_criter < 0:
                raise ValueError("delta_criter must be positive")

        if likelihood_ratio:
            if likelihood_ratio <= 0 or likelihood_ratio >= 1:
                raise ValueError("likelihood_ratio must be in (0,1)")

        self.parent = parent
        self.n_init = n_init
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
        
    def fit(self, X, adj=None):
        self.fit_predict(X)
        return self

    def predict(self, X, adj=None):
        check_is_fitted(self, ["model_"], all_or_any=all)
        labels = self.fit_predict(X)
        return labels
    
    def fit_predict(self, X, level=None, adj=None):
        X = check_array(
            X, dtype=[np.float64, np.float32],  ensure_min_samples=1
        )
        if level:
            if not isinstance(level, int):
                raise TypeError("level must be an int")

        self.X = X
        self.adj = adj

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

        labels = self._to_labels()
        if level:
            labels = labels[:, level]

        self.predictions = labels
        return labels

    def _get_next_node(self):
        leaves = self.root.leaves
        current_node = []

        for leaf in leaves:
            if (len(leaf.X) >= self.max_components and
                    len(leaf.X) >= self.min_split and
                    leaf.depth < self.max_level):
                if not hasattr(leaf, "k_"):
                    current_node = leaf
                    break
        return current_node

    def _cluster_and_decide(self):
        X = self.X
        if self.cluster_method == "GMM":
            cluster = AutoGMMCluster(
                min_components=1, max_components=self.max_components, **self.cluster_kws
            )
            pred = cluster.fit_predict(X)
            model = cluster.model_
            criter = cluster.criter_
            lik = model.score(X)
            k = cluster.n_components_

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
                    p = chi2.sf(LR, k-1)
                    # TODO: maybe set a default p-value threshold if do LR test?
                    if p < self.likelihood_ratio:
                        k = 1
                # TODO: maybe add similar tests for Kmeans?

        elif self.cluster_method == "Kmeans":
            cluster = KMeansCluster(
                max_clusters=self.max_components, **self.cluster_kws
            )
            pred = cluster.fit_predict(X)
            model = cluster.model_
            k = cluster.n_clusters_
            criter = cluster.silhouette_
            
        elif self.cluster_method == "Spherical-Kmeans":
            # cluster = SphericalKMeans(
            #     n_clusters=self.max_components, **self.cluster_kw
            # )
            pass

        results = {
            "model": model,
            "pred": pred,
            "criter": criter,
            "k": k,
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

                if self.adj:
                    new_adj = self.adj[np.ix_(inds, inds)]
                    new_x = self._embed(new_adj)

                RecursiveCluster(
                    selection_criteria=self.selection_criteria,
                    cluster_method=self.cluster_method,
                    n_init=self.n_init,
                    parent=self,
                    max_components=self.max_components,
                    min_split=self.min_split,
                    cluster_kws=self.cluster_kws,
                    delta_criter=self.delta_criter,
                    likelihood_ratio=self.likelihood_ratio,
                )
                self.children[-1].X = new_x
                self.children[-1].inds = inds

                if self.adj:
                    self.children[-1].adj = new_adj
                else:
                    self.children[-1].adj = self.adj

    def _embed(self, graph):
        raise NotImplementedError()

    def _to_labels(self):
        n_levels = self.height
        n_sample = len(self.X)
        labels = np.zeros((n_sample, n_levels))
        labels[:, 0] = self.pred_
        children = [node for node in LevelOrderIter(self)][1:]

        for c in range(len(children)):
            level = children[c].depth
            if level < n_levels and hasattr(children[c], "pred_"):
                path = children[c].path[1:]
                X_inds = np.arange(n_sample)
                for node in range(len(path)):
                    X_inds = X_inds[path[node].inds]
                labels[X_inds, level] = children[c].pred_

        for level in range(1, n_levels):
            # to avoid updated labels accidentally matching old ones
            new_labels = -labels[:, :level+1]-1
            uni_labels = np.unique(new_labels, axis=0)
            for ul in range(len(uni_labels)):
                uni_labels_inds = np.where(
                    (uni_labels[:, None, :] == new_labels).all(2)[ul]
                    )[0]
                labels[uni_labels_inds, level] = ul

        return labels


class _RecursiveGraphCluster(RecursiveCluster):
    def __init__(
        self,
        selection_criteria=None,
        cluster_method="GMM",
        embed_method="ASE",
        n_init=1,
        parent=None,
        min_components=1,
        max_components=10,
        n_components=None,
        cluster_kws={},
        embed_kws={},
        min_split=1,
        max_level=20,
        delta_criter=None,
        likelihood_ratio=None,
        loops=False,
    ):
        _check_common_inputs(
            min_components, max_components, cluster_kws, embed_kws
        )

        if embed_method not in ["ASE", "LSE"]:
            msg = "clustering method must be one of {ASE, LSE}"
            raise ValueError(msg)

        if n_components:
            # TODO: assume every subgraph embedded to "n_components" if given?
            if not isinstance(n_components, int):
                raise TypeError("n_components must be an int")

        RecursiveCluster.__init__(
            self,
            selection_criteria=selection_criteria,
            cluster_method=cluster_method,
            n_init=n_init,
            parent=parent,
            min_components=min_components,
            max_components=max_components,
            cluster_kws=cluster_kws,
            min_split=min_split,
            max_level=max_level,
            delta_criter=delta_criter,
            likelihood_ratio=likelihood_ratio,
        )

        self.n_components = n_components
        self.embed_method = embed_method
        self.embed_kws = embed_kws
        self.loops = loops

    def fit(self, graph):
        self.fit_predict(graph)
        return self

    def fit_predit(self, graph):
        graph = import_graph(graph)
        if not self.loops:
            graph = remove_loops(graph)
     
        graph = augment_diagonal(graph)
        embed_graph = pass_to_ranks(graph)
        latent = self._embed(embed_graph)
        if isinstance(latent, tuple):
            latent = np.concatenate(latent, axis=1)

        labels = super().fit_predict(X=latent, adj=embed_graph)
        self.predictions = labels

        return labels

    def predict(self, graph):
        check_is_fitted(self, ["model_"], all_or_any=all)
        return self.predictions

    def _embed(self, graph):
        if self.n_components:
            n_components = self.n_components
        else:
            # TODO: not embedding subgraphs onto the same dim
            # in case some subgraph contains very few nodes
            n_components = None

        if self.embed_method == "ASE":
            embedder = AdjacencySpectralEmbed(
                n_components=n_components, **self.embed_kws
            )
            embed = embedder.fit_transform(graph)
        elif self.embed_method == "LSE":
            embedder = LaplacianSpectralEmbed(
                n_components=n_components, **self.embed_kws
            )
            embed = embedder.fit_transform(graph)

        return embed


class HSBMEstimator(_RecursiveGraphCluster, BaseGraphEstimator):
    def __init__(
        self,
        selection_criteria=None,
        cluster_method="GMM",
        embed_method="ASE",
        n_init=1,
        parent=None,
        min_components=1,
        max_components=10,
        n_components=None,
        cluster_kws={},
        embed_kws={},
        min_split=1,
        max_level=20,
        delta_criter=None,
        likelihood_ratio=None,
        directed=False,
        loops=False,
        reembed=False,
    ):
        _RecursiveGraphCluster.__init__(
            self,
            selection_criteria=selection_criteria,
            cluster_method=cluster_method,
            n_init=n_init,
            parent=parent,
            min_components=min_components,
            max_components=max_components,
            cluster_kws=cluster_kws,
            min_split=min_split,
            max_level=max_level,
            delta_criter=delta_criter,
            likelihood_ratio=likelihood_ratio,
            n_components=n_components,
            embed_kws=embed_kws,
        )

        BaseGraphEstimator.__init__(self, directed=directed, loops=loops)

        self.reembed = reembed

    def fit(self, X, y=None):
        if y is None:
            if self.reembed is True:
                y = _RecursiveGraphCluster.fit_predict(self, X)
            else:
                y = RecursiveCluster.fit_predict(self, X)
     
        _, y = check_X_y(X, y, multi_output=True)

        if max(y[0]) == 0:
            warnings.warn("only 1 cluster predicted at the first level")
        
        n_levels = y.shape[1]
      
        self.block_weights_ = np.empty(n_levels, dtype=object)
        self.block_p_ = np.empty(n_levels, dtype=object)
        self.p_mat_ = np.empty(n_levels, dtype=object)

        for i in range(n_levels):
            single_label = y[:, i]
            _, counts = np.unique(single_label, return_counts=True)
            self.block_weights_[i] = counts / X.shape[0]
            block_vert_inds, block_inds, block_inv = _get_block_indices(single_label)
            block_p = _calculate_block_p(X, block_inds, block_vert_inds)

            if self.reembed is True:
                if not self.directed:
                    block_p = symmetrize(block_p)
            self.block_p_[i] = block_p

            p_mat = _block_to_full(block_p, block_inv, X.shape)
            if self.reembed is True:
                if not self.loops:
                    p_mat = remove_loops(p_mat)
            self.p_mat_[i] = p_mat

        return self
