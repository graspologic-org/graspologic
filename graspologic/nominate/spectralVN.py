# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Union
from abc import abstractmethod
from .base import BaseVN
from ..embed import BaseEmbed
from ..embed import AdjacencySpectralEmbed as ase
import numpy as np
from scipy.spatial import distance
from scipy.stats import mode

class BaseSpectralVN(BaseVN):
    def __init__(self, multigraph: bool, embedding: np.ndarray, embeder: Union[str, BaseEmbed], mode: str):
        super().__init__(multigraph=multigraph)
        self.embedding = embedding
        if self.embedding is None:
            if issubclass(type(embeder), BaseEmbed):
                self.embeder = embeder
            elif embeder == 'ASE':
                self.embeder = ase()
            else:
                raise TypeError
        self.seed = None
        self._attr_labels = None
        self.unique_att = None
        self.mode = mode

    def _pairwise_dist(self, y: np.ndarray, metric='euclidean') -> np.ndarray:
        # wrapper for scipy's cdist function
        # y should give indexes
        y_vec = self.embedding[y[:, 0]]
        dist_mat = distance.cdist(self.embedding, y_vec, metric=metric)
        return dist_mat

    def _embed(self, X: np.ndarray):
        # ensure X matches required dimensions for single and multigraph
        if self.multigraph and (len(X.shape) < 3 or X.shape[0] <= 1):
            raise IndexError("Argument must have dim 3")
        if not self.multigraph and len(X.shape) != 2:
            if len(X.shape) == 3 and X.shape[0] <= 1:
                X = X.reshape(X.shape[1], X.shape[2])
            else:
                raise IndexError("Argument must have dim 2")

        # Embed graph if embedding not provided
        if self.embedding is None:
            self.embedding = self.embeder.fit_transform(X)

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Constructs the embedding if needed.
        Parameters
        ----------
        X
        y: List of seed vertex indices, OR List of tuples of seed vertex indices and associated attributes.

        Returns
        -------

        """
        X = np.array(X)
        if self.embedding is None:
            self._embed(X)

        # detect if y is attributed
        y = np.array(y)
        if np.ndim(y) < 2:
            y = y.reshape(1, 2)
        else:
            y = y.reshape(-1, 2)
        self._attr_labels = y[:, 1]
        self.seed = y[:, 0]
        self.unique_att = np.unique(self._attr_labels)

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def fit_transform(self, X, y=None):
        pass


class SpectralVertexNominator(BaseSpectralVN):

    def __init__(self, multigraph: bool = False,
                 embedding: np.ndarray = None,
                 embeder: Union[str, BaseEmbed] = 'ASE',
                 mode: str = 'single_vertex'):
        super(SpectralVertexNominator,
              self).__init__(multigraph=multigraph,
                             embedding=embedding,
                             embeder=embeder,
                             mode=mode)
        self.distance_matrix = None

    def fit(self, X, y):
        self._fit(X, y)
        self.distance_matrix = self._pairwise_dist(y)

    def _knn_simple_predict(self):
        '''
        Simplest possible methdod, doesn't consider attributes.
        If multiple seed vertices are provied, provides the top
        nominations for each individual seed.
        Returns
        -------

        '''
        ordered = self.distance_matrix.argsort(axis=1)
        sorted_dists = self.distance_matrix[np.arange(ordered.shape[0], ordered.T)].T
        return ordered, sorted_dists

    def _knn_weighted_predict(self, out, k=5):
        '''
        Nominate vertex based on distance from the k nearest neighbors of each class.
        The default decision function is sum(dist to each neighbor of class c) / (number_such_neighbors)^2.
        This is a good metric becuase it accounts for both number of neighbors from a class and their respective
        distances. However, assumes that all possible attributes are represented in the seed population.

        Parameters
        ----------
        out

        Returns
        -------

        '''
        ordered = self.distance_matrix.argsort(axis=1)
        sorted_dists = self.distance_matrix[np.arange(ordered.shape[0]), ordered.T].T
        atts = self._attr_labels[ordered[:, :k]]  # label for the nearest 5 seeds for each vertex
        pred_weights = np.empty(
            (atts.shape[0], self.unique_att.shape[0]))  # use this array for bin counts as well to save space
        for i in range(self.unique_att.shape[0]):
            pred_weights[:, i] = np.count_nonzero(atts == self.unique_att[i], axis=1)
            inds = np.argwhere(atts == self.unique_att[i])
            place_hold = np.empty(atts.shape)
            place_hold[:] = np.NaN
            place_hold[inds[:, 0], inds[:, 1]] = sorted_dists[inds[:, 0], inds[:, 1]]
            pred_weights[:, i] = np.nansum(place_hold, axis=1) / np.power(pred_weights[:, i], 2)
        if out == 'best_preds':
            best_pred_inds = np.nanargmin(pred_weights, axis=1)
            best_pred_weights = pred_weights[np.arange(pred_weights.shape[0]), best_pred_inds]
            vert_order = np.argsort(best_pred_weights, axis=0)
            att_preds = self.unique_att[best_pred_inds[vert_order]]
            prediction = np.concatenate((vert_order.reshape(-1, 1), att_preds.reshape(-1, 1)), axis=1)
            return prediction, pred_weights[vert_order]
        elif out == 'per_attribute':
            vert_order = np.empty(pred_weights.shape, dtype=np.int)
            for i in range(pred_weights.shape[1]):
                pred_weights[np.argwhere(np.isnan(pred_weights[:, i])), i] = np.nanmax(pred_weights[:, i])
                vert_order[:, i] = np.argsort(pred_weights[:, i])
            return vert_order, pred_weights[vert_order]

    def predict(self, out="best_preds"):
        if self.mode == 'single_vertex':
            return self._knn_simple_predict()
        elif self.mode == 'knn-weighted':
            return self._knn_weighted_predict(out)
        else:
            raise KeyError("no such mode " + str(self.mode))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.predict()


class SpectralClusterVertexNominator(BaseSpectralVN):
    def __init__(self, multigraph: bool = False,
                 embedding: np.ndarray = None,
                 embeder: Union[str, BaseEmbed] = 'ASE',
                 mode: str = 'single_vertex'):
        super(SpectralClusterVertexNominator,
              self).__init__(multigraph=multigraph,
                             embedding=embedding,
                             embeder=embeder,
                             mode=mode)
        self.clf = None

    def fit(self, X, y):
        """
        Unsupervised kmeans spectral nomination, as described
        by Fishkind et. al. in Vertex Nomination for Membership Prediction.
        Has the advantage of being able to identify attributes not represented
        in seed population, however by default will select number of clusters
        based on number of unique attributes in given seed.
        Returns
        -------

        """
        from sklearn.cluster import KMeans
        self._fit(X, y)
        self.clf = KMeans(n_clusters=self.unique_att.shape[0])
        self.clf.fit(self.embedding)

    def _cluster_map(self, y_hat):
        clusters = np.sort(np.unique(y_hat))
        metric_arr = np.zeros((clusters.shape[0], clusters.shape[0]), dtype=np.float32)
        for i in range(clusters.shape[0]):
            att_ind = np.argwhere(y_hat == clusters[i]).reshape(-1)
            _, seed_arg, _ = np.intersect1d(self.seed, att_ind, return_indices=True)
            temp_att = np.concatenate((self._attr_labels[seed_arg], self.unique_att), axis=0)  #ensure all rep'ed in count
            id, counts = np.unique(temp_att, return_counts=True)
            ind = np.argsort(id)
            counts = counts[ind]  # leave some smoothing bias here to prevent nan slice
            metric_arr[i] = counts / temp_att.shape[0]  # percent each label att
        map = np.zeros(clusters.shape[0], dtype=np.int)
        # assign cluster mapping greedily in percent match
        for i in range(clusters.shape[0]):
            best = np.unravel_index(np.nanargmax(metric_arr), metric_arr.shape)
            og_att = np.sort(self.unique_att)[best[1]]
            map[best[0].astype(int)] = og_att.astype(int)
            metric_arr[best[0]] = np.nan
            metric_arr[:, best[1]] = np.nan
        return map

    def predict(self, out='per_attribute'):
        y_hat = self.clf.predict(self.embedding)
        clust_to_att = self._cluster_map(y_hat)
        centered_map = clust_to_att - np.min(clust_to_att)  # allows to order prediction as they were provided
        clust_dists = self.clf.transform(self.embedding)
        att_preds = np.empty(clust_dists.shape, dtype=np.int)
        for i in range(clust_dists.shape[1]):
            sort_inds = np.argsort(clust_dists[:, i])
            att_preds[:, i] = sort_inds
            clust_dists[:, i] = clust_dists[sort_inds, i]
        att_preds = att_preds.T[centered_map].T
        clust_dists = clust_dists.T[centered_map].T
        return att_preds, np.sort(clust_to_att), clust_dists


