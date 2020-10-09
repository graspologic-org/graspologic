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


# STATIC METHODS #

def _make_2d(cls, arr):
    arr = np.array(arr, dtype=np.int)
    if np.ndim(arr) < 2:
        arr = np.concatenate((arr, np.zeros(arr.shape[0])))
        arr = arr.reshape(-1, 2)
    else:
        arr = arr.reshape(-1, 2)
    return arr


class SpectralVertexNominator(BaseVN):
    """

    """

    def __init__(
            self,
            multigraph: bool = False,
            embedding: np.ndarray = None,
            embeder: Union[str, BaseEmbed] = "ASE",
            mode: str = "single_vertex",
    ):
        super().__init__(multigraph=multigraph)
        self.embedding = embedding
        if self.embedding is None:
            if issubclass(type(embeder), BaseEmbed):
                self.embeder = embeder
            elif embeder == "ASE":
                self.embeder = ase()
            else:
                raise TypeError
        self.seed = None
        self._attr_labels = None
        self.unique_att = None
        self.mode = mode
        self.distance_matrix = None

    def _pairwise_dist(self, y: np.ndarray, metric="euclidean") -> np.ndarray:
        # wrapper for scipy's cdist function
        # y should give indexes
        y_vec = self.embedding[y[:, 0].astype(np.int)]
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

    def fit(self, X, y):
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

        self._attr_labels = y[:, 1]
        self.seed = y[:, 0]
        self.unique_att = np.unique(self._attr_labels)

    def _knn_simple_predict(self):
        """
        Simplest possible methdod, doesn't consider attributes.
        If multiple seed vertices are provied, provides the top
        nominations for each individual seed.
        Returns
        -------

        """
        ordered = self.distance_matrix.argsort(axis=0)
        sorted_dists = np.zeros(ordered.shape)
        for i in range(ordered.shape[1]):
            sorted_dists[:, i] = self.distance_matrix[ordered[:, i], i].reshape(-1)
        return ordered, sorted_dists, np.zeros(1)

    def _knn_weighted_predict(self, out, k=5):
        """
        Nominate vertex based on distance from the k nearest neighbors of each class.
        The default decision function is sum(dist to each neighbor of class c) / (number_such_neighbors)^2.
        This is a good metric because it accounts for both number of neighbors from a class and their respective
        distances. However, assumes that all possible attributes are represented in the seed population.

        Parameters
        ----------
        out

        Returns
        -------

        """
        ordered = self.distance_matrix.argsort(axis=1)
        sorted_dists = self.distance_matrix[np.arange(ordered.shape[0]), ordered.T].T
        atts = self._attr_labels[
            ordered[:, :k]
        ]  # label for the nearest 5 seeds for each vertex
        pred_weights = np.empty(
            (atts.shape[0], self.unique_att.shape[0])
        )  # use this array for bin counts as well to save space
        for i in range(self.unique_att.shape[0]):
            pred_weights[:, i] = np.count_nonzero(atts == self.unique_att[i], axis=1)
            inds = np.argwhere(atts == self.unique_att[i])
            place_hold = np.empty(atts.shape)
            place_hold[:] = np.NaN
            place_hold[inds[:, 0], inds[:, 1]] = sorted_dists[inds[:, 0], inds[:, 1]]
            pred_weights[:, i] = np.nansum(place_hold, axis=1) / np.power(
                pred_weights[:, i], 2
            )
        if out == "best_preds":
            best_pred_inds = np.nanargmin(pred_weights, axis=1)
            best_pred_weights = pred_weights[
                np.arange(pred_weights.shape[0]), best_pred_inds
            ]
            vert_order = np.argsort(best_pred_weights, axis=0)
            att_preds = self.unique_att[best_pred_inds[vert_order]]
            prediction = np.concatenate(
                (vert_order.reshape(-1, 1), att_preds.reshape(-1, 1)), axis=1
            )
            return prediction, pred_weights[vert_order], self.unique_att
        elif out == "per_attribute":
            vert_order = np.empty(pred_weights.shape, dtype=np.int)
            for i in range(pred_weights.shape[1]):
                pred_weights[np.argwhere(np.isnan(pred_weights[:, i])), i] = np.nanmax(
                    pred_weights[:, i]
                )
                vert_order[:, i] = np.argsort(pred_weights[:, i])
            return vert_order, pred_weights[vert_order], self.unique_att

    def predict(self, out="best_preds"):
        if self.mode == "single_vertex":
            return self._knn_simple_predict()
        elif self.mode == "knn_weighted":
            return self._knn_weighted_predict(out)
        else:
            raise KeyError("no such mode " + str(self.mode))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.predict()
