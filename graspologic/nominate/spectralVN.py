# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Union, Tuple
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

    def _pairwise_dist(self, y: np.ndarray, metric: str = "euclidean") -> np.ndarray:
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

    def _knn_predict(self, k: np.uint16 = 5, neighbor_function: str = "sum_inverse_distance") -> Tuple[np.ndarray, np.ndarray]:
        """
        Nominate vertex based on distance from the k nearest neighbors of each class.
        The default decision function is sum(dist to each neighbor of class c) / (number_such_neighbors)^2.
        This is a good metric because it accounts for both number of neighbors from a class and their respective
        distances. However, assumes that all possible attributes are represented in the seed population.

        Parameters
        ----------
        k : Number of neighbors to consider in nearest neighbors classification
        neighbor_function : method for determining class membership based on neighbors
            options
            ------
            sum_inverse_distance :  Simplest weighted knn method, works well in the VN context because
                                    it generates a natural ordering for each vertex on each attribute represented
                                    in the seed set. For each attribute, nomination is ordered by
                                    sum of the inverse of distances to the k nearest neighbors belonging
                                    to that attribute.
        Returns
        -------
        An tuple of two np.ndarrays, each of shape(number_vertices, number_attributes_in_seed).
        The array at index 0 is the nomination list, for each attribute column, the rows are indexes

        """
        num_att = self.unique_att.shape[0]

        ordered = self.distance_matrix.argsort(axis=1)
        sorted_dists = self.distance_matrix[np.arange(ordered.shape[0]), ordered.T].T
        atts = self._attr_labels[
            ordered[:, :k]
        ]  # label for the nearest k seeds for each vertex

        # could avoid mem penalty by also broadcasting here, but is slightly slower and reduces code clarity
        nd_buffer = np.tile(atts, (k, 1, 1)).astype(np.float32)
        inds = np.argwhere(nd_buffer == self.unique_att[:, np.newaxis, np.newaxis])

        # nans are a neat way to operate on attributes individually
        nd_buffer[:] = np.NaN
        nd_buffer[inds[:, 0], inds[:, 1], inds[:, 2]] = sorted_dists[
            inds[:, 1], inds[:, 2]
        ]

        # weighting function, outer inverse for consistency (e.g. higher rank has smaller distance metric)
        if neighbor_function == "sum_inverse_distance":
            pred_weights = np.power(np.nansum(np.power(nd_buffer, -1), axis=2), -1).T
        else:
            raise KeyError

        nan_inds = np.argwhere(np.isnan(pred_weights))
        pred_weights[nan_inds[:, 0], nan_inds[:, 1]] = np.nanmax(pred_weights)
        vert_order = np.argsort(pred_weights, axis=0)

        return vert_order, pred_weights[vert_order]

    def fit(self, X: np.ndarray, y: np.ndarray):
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
        self.distance_matrix = self._pairwise_dist(y)

    def predict(self, out: str = "best_preds") -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == "knn_weighted":
            return self._knn_predict()
        else:
            raise KeyError("No such mode " + str(self.mode))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(X, y)
        return self.predict()
