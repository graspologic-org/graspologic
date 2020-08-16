import numpy as np
import pandas as pd
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

    if embed_kws is not None:
        if not isinstance(embed_kws, dict):
            raise TypeError("embed_kws must be a dict")

    if not isinstance(cluster_kws, dict):
        raise TypeError("cluster_kws must be a dict")


class RecursiveCluster(NodeMixin, BaseEstimator):
    def __init__(
        self,
        selection_criteria_GMM="bic",
        cluster_method="GMM",
        root_inds=None,
        n_init=1,
        parent=None,
        min_components=1,
        max_components=10,
        cluster_kws={},
        # labels=None,  # not expecting label_init?
        min_split=1,
        reembed=False,
    ):

        _check_common_inputs(
            min_components, max_components, cluster_kws, embed_kws=None
        )

        if cluster_method not in ["GMM", "KMeans"]:
            msg = "clustering method must be one of {GMM, Kmeans}"
            raise ValueError(msg)

        self.parent = parent
        self.n_init = n_init
        self.min_components = min_components
        self.max_components = max_components
        self.cluster_method = cluster_method
        self.min_split = min_split
        self.selection_criteria_GMM = selection_criteria_GMM
        self.cluster_kws = cluster_kws
        self.min_split = min_split
        self.root_inds = root_inds
        self.reembed = reembed

    def fit(self, X, adj=None):
        self.fit_predict(X)
        return self

    def predict(self, X, adj=None):
        check_is_fitted(self, ["model_"], all_or_any=all)
        return self.predictions

    def fit_predict(self, X, adj=None):
        X = check_array(
            X, dtype=[np.float64, np.float32],  ensure_min_samples=1
        )
        self.X = X
        self.adj = adj

        labels = pd.DataFrame(range(len(self.X)))
        labels.columns = ["inds"]
        self.labels = labels
        if self.root_inds is None:
            self.root_inds = labels["inds"]

        if self.max_components > X.shape[0]:
            msg = "max_components must be >= n_samples, but max_components = "
            msg += "{}, n_samples = {}".format(self.max_components, X.shape[0])
            raise ValueError(msg)
        
        while True:
            current_node = self._get_next_node()
            if current_node:
                current_node._fit_cluster()
                current_node._select_model()
            else:
                break

        labels = self._collect_labels()
        self.predictions = labels
        return labels

    def _get_next_node(self):
        leaves = self.root.leaves
        current_node = []

        for leaf in leaves:
            if (len(leaf.X) >= self.max_components and
                    len(leaf.X) >= self.min_split):
                if not hasattr(leaf, "k_"):
                    current_node = leaf
                    break
        return current_node

    def _fit_cluster(self):
        X = self.X
        if self.cluster_method == "GMM":
            cluster = AutoGMMCluster(
                min_components=1, max_components=self.max_components, **self.cluster_kws
            )
            pred = cluster.fit_predict(X)
            model = cluster.model_
            bic = model.bic(X)
            lik = model.score(X)
            k = cluster.n_components_

        elif self.cluster_method == "Kmeans":
            cluster = KMeansCluster(
                max_clusters=self.max_components, **self.cluster_kws
            )
            cluster.fit(X)
            model = cluster.model_
            k = cluster.n_clusters_
            pred = cluster.predict(X)

        results = {
            "bic": -bic,
            "lik": lik,
            "k": k,
            "model": model,
            "pred": pred,
        }
        self.results_ = results

    def _select_model(self, k=None):
        model = self.results_["model"]
        pred = self.results_["pred"]
        k = self.results_["k"]

        self.k_ = k
        self.children = []

        if k > 1:
            self.model_ = model
            self.pred_ = pred
            root_labels = self.root.labels

            pred_name = f"{self.depth + 1}_pred"
            if pred_name not in root_labels.columns:
                root_labels[pred_name] = ""
            root_labels.loc[self.root_inds, pred_name] = pred
            uni_labels = np.unique(pred)

            for ul in uni_labels:
                new_labels = root_labels[
                    (root_labels[pred_name] == ul)
                    & (root_labels.index.isin(self.root_inds.index))
                ]
                new_root_inds = new_labels["inds"]
                new_x = self.root.X[new_root_inds]

                if self.adj is not None:
                    new_adj = self.root.adj[
                        np.ix_(new_root_inds, new_root_inds)
                    ]
                    if self.reembed is True:
                        new_x = self._embed(new_adj, reembedding=True)

                RecursiveCluster(
                    selection_criteria_GMM=self.selection_criteria_GMM,
                    cluster_method=self.cluster_method,
                    root_inds=new_root_inds,
                    n_init=self.n_init,
                    parent=self,
                    max_components=self.max_components,
                    min_split=self.min_split,
                    cluster_kws=self.cluster_kws,
                    reembed=self.reembed,
                )
                self.children[-1].X = new_x

                if self.adj is not None:
                    self.children[-1].adj = new_adj
                else:
                    self.children[-1].adj = self.adj
            
    # TODO: maybe add tuning params like delta_bic or likelihood ratio
    # for model selection

    def _embed(self, graph, reembedding=False):
        raise NotImplementedError()

    def _collect_labels(self):
        labels = self.root.labels.replace("", -1)
        n_levels = self.height
        preds = []
        for i in range(1, n_levels+1):
            preds.append(f"{i}_pred")
            labels_pred = labels.loc[:, preds]
            uni_labels = np.unique(labels_pred.values, axis=0)
            labels[f"lvl{i}_labels"] = ''
            for j in range(len(uni_labels)):
                uni_labels_inds = np.where(
                    (uni_labels[:, None, :] == labels_pred.values).all(2)[j]
                    )[0]
                labels.loc[uni_labels_inds, f"lvl{i}_labels"] = j
        mask = labels.columns.str.endswith(('pred'))
        labels = labels.loc[:, ~mask]
        # add lvl0_labels in case no split is prediced by agmm at first level
        labels.insert(0, "lvl0_labels", np.zeros(len(labels)).astype(int))
        labels = labels.drop(columns=['inds'])

        return labels


class HSBMEstimator(RecursiveCluster, BaseGraphEstimator):
    def __init__(
        self,
        selection_criteria_GMM="bic",
        cluster_method="GMM",
        embed_method="ASE",
        root_inds=None,
        n_init=1,
        parent=None,
        min_components=1,
        max_components=10,
        n_components=None,
        n_components_reembed=None,
        cluster_kws={},
        embed_kws={},
        min_split=1,
        reembed=False,
        directed=False,
        loops=False,
        subgraph_similarity=False,
        subgraph_similarity_lvl=None,
    ):
        _check_common_inputs(
            min_components, max_components, cluster_kws, embed_kws
        )

        if cluster_method not in ["GMM", "KMeans"]:
            msg = "clustering method must be one of {GMM, Kmeans}"
            raise ValueError(msg)

        if embed_method not in ["ASE", "LSE"]:
            msg = "clustering method must be one of {ASE, LSE}"
            raise ValueError(msg)

        RecursiveCluster.__init__(
            self,
            selection_criteria_GMM=selection_criteria_GMM,
            cluster_method=cluster_method,
            root_inds=root_inds,
            n_init=n_init,
            parent=parent,
            min_components=min_components,
            max_components=max_components,
            cluster_kws=cluster_kws,
            min_split=min_split,
            reembed=reembed,
        )

        BaseGraphEstimator.__init__(self, directed=directed, loops=loops)

        self.n_components = n_components
        self.n_components_reembed = n_components_reembed
        self.embed_method = embed_method
        self.embed_kws = embed_kws
        self.subgraph_similarity = subgraph_similarity
        self.subgraph_similarity_lvl = subgraph_similarity_lvl

    def _estimate_assignments(self, graph):
        graph = augment_diagonal(graph)
        embed_graph = pass_to_ranks(graph)
        latent = self._embed(embed_graph, reembedding=False)
        if isinstance(latent, tuple):
            latent = np.concatenate(latent, axis=1)

        vertex_assignments = super().fit_predict(latent, embed_graph)
        self.vertex_assignments_ = vertex_assignments

    def fit(self, graph, y=None):
        graph = import_graph(graph)
        if not self.loops:
            graph = remove_loops(graph)
     
        if y is None:
            self._estimate_assignments(graph)
            y = self.vertex_assignments_
     
        _, y = check_X_y(graph, y, multi_output=True)
        if y.shape[1] == 1:
            # warnings.warn("predicted number of components is 1")
            # msg = "no blocks found: predicted number of components is 1"
            # raise ValueError(msg)
            pass
        
        y = y[:, 1:]
        n_levels = y.shape[1]
      
        self.block_weights_ = np.empty(n_levels, dtype=object)
        self.block_p_ = np.empty(n_levels, dtype=object)
        self.p_mat_ = np.empty(n_levels, dtype=object)

        for i in range(n_levels):
            label_single = y[:, i]
            _, counts = np.unique(label_single, return_counts=True)
            self.block_weights_[i] = counts / graph.shape[0]
            block_vert_inds, block_inds, block_inv = _get_block_indices(label_single)
            block_p = _calculate_block_p(graph, block_inds, block_vert_inds)

            if not self.directed:
                block_p = symmetrize(block_p)
            self.block_p_[i] = block_p

            p_mat = _block_to_full(block_p, block_inv, graph.shape)
            if not self.loops:
                p_mat = remove_loops(p_mat)
            self.p_mat_[i] = p_mat

        return self

    def _embed(self, graph, reembedding):
        # reembedding = False for 1st-time embed before clustering
        # reembedding = True for embed during clustering if self.reembed=True
        n_components = self._embed_components(reembedding=reembedding)

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

    def _embed_components(self, reembedding):
        if reembedding is False:
            if self.n_components is not None:
                n_components = self.n_components
            else:
                n_components = None
        else:
            if self.n_components_reembed is not None:
                # TODO: check this input
                # (should it be one int or a list of them?)
                n_components = self.n_components_reembed
            else:
                # TODO: calculate max_dim at current level
                # should be easier if do clustering lvl by lvl in RC
                pass
        return n_components

# TODO: subgraph dissimilarity
# if self.subgraph_similarity is True:
#     # TODO: check self.subgraph_similarity_lvl
#     # and then compute at that lvl
#     subgraph_dissimilarities = _compute_subgraph_dissimilarities(
#         subgraph_latents, sub_inds, self.bandwidth
#     )
