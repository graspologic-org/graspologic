# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Union, Tuple
from .base import BaseVN
from ..embed import BaseSpectralEmbed
from ..embed import AdjacencySpectralEmbed as ase, LaplacianSpectralEmbed as lse
import numpy as np
from scipy.spatial import distance


class SpectralVertexNominator(BaseVN):
    """
    Class for spectral vertex nomination on a single graph.

    Given a graph G=(V,E) and a subset of V called S (the "seed"),
    Single Graph Vertex Nomination is the problem of ranking all V
    in order of relation to members of S.
    Spectral Vertex Nomination solves this problem by embedding G
    into a low dimensional euclidean space using any of the embedding
    algorithms (:ref:`tutorials <embed_tutorials>`),
    and then generating a nomination list by some distance based algorithm.
    In the simplest case, for each seed vertex u, the other vertices are
    ranked in order of euclidean distance from u.
    There can be both attributed and unattributed cases. In the unattributed
    case, we treat each seed vertex individually and rank all vertices by distance.
    In the attributed case, subsets of the seed vertices share some attribute we
    care about. We want to rank the other vertices by the likelihood that they
    also share that attribute. Note that the unattributed problem is simply the
    attributed problem when each seed vertex has a unique attribute.
    SVN algorithms in general make the assumption that vertexes
    are likely to be related in a way that is of interest if they are close to
    each other in an embedding. This is a somewhat simplified notion of relatedness,
    and is not appropriate for all Vertex Nomination problems.

    Parameters
    ----------
    embedding: np.ndarray, default = None
        A pre-calculated embedding may be provided, in which case
        it will be used for vertex nomination instead of embedding
        the adjacency matrix using embeder.
    embeder: str or BaseEmbed, default = 'ASE'
        May provide either a embed object or a string indicating
        which embedding method to use, which may be either
        "ASE" for :py:class:`~graspologic.embed.AdjacencySpectralEmbed` or
        "LSE" for :py:class:`~graspologic.embed.LaplacianSpectralEmbed`.
    persistent : bool, default = True
        If ``False``, future calls to fit will overwrite an existing embedding. Must be ``True``
        if an embedding is provided.


    Attributes
    ----------
    embedding : np.ndarray
        The spectral embedding of the graph that spectral nomination will be preformed on.
    embeder : :py:class:`~graspologic.embed.BaseEmbed`
        The embed object to be used to compute the embedding.
    attr_labels : np.ndarray
        The attributes of the vertices in the seed (parameter 'y' for fit).
        Shape is the number of seed vertices.
    unique_att : np.ndarray
        Each unique attribute represented in the seed. One dimensional.
    distance_matrix : np.ndarray
        The euclidean distance from each seed vertex to each vertex.
        Shape is ``(number_vertices, number_unique_attributes)`` if attributed
        or Shape is ``(number_vertices, number_seed_vertices)`` if unattributed.
    persistent : bool
        If ``False``, future calls to fit will overwrite an existing embedding. Must be ``True``
        if an embedding is provided.

    References
    ----------
    Fishkind, D. E.; Lyzinski, V.; Pao, H.; Chen, L.; Priebe, C. E. Vertex nomination schemes for membership prediction.
    Ann. Appl. Stat. 9 2015. https://projecteuclid.org/euclid.aoas/1446488749

    Jordan Yoder, Li Chen, Henry Pao, Eric Bridgeford, Keith Levin, Donniell E. Fishkind, Carey Priebe, Vince Lyzinski,
    Vertex nomination: The canonical sampling and the extended spectral nomination schemes,
    Computational Statistics & Data Analysis, Volume 145, 2020.
    http://www.sciencedirect.com/science/article/pii/S0167947320300074

    """

    def __init__(
        self,
        embedding: np.ndarray = None,
        embeder: Union[str, BaseSpectralEmbed] = "ASE",
        persistent: bool = True,
    ):
        super().__init__(multigraph=False)
        self.embedding = embedding
        if self.embedding is None or not persistent:
            if issubclass(type(embeder), BaseSpectralEmbed):
                self.embeder = embeder
            elif embeder == "ASE":
                self.embeder = ase()
            elif embeder == "LSE":
                self.embeder = lse()
            else:
                raise TypeError
        elif np.ndim(embedding) != 2:
            raise IndexError("embedding must have dimension 2")
        self.persistent = persistent
        self.attr_labels = None
        self.unique_att = None
        self.distance_matrix = None

    @staticmethod
    def _make_2d(arr: np.ndarray) -> np.ndarray:
        # ensures arr is two or less dimensions.
        # if 1d, adds unique at each index on
        # the second dimension.
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError("Argument must be of type int")
        arr = np.array(arr, dtype=np.int)
        if np.ndim(arr) > 2 or (arr.ndim == 2 and arr.shape[1] > 2):
            raise IndexError("Argument must have shape (n) or (n, 1) or (n, 2).")
        elif np.ndim(arr) == 1 or arr.shape[1] == 1:
            arr = arr.reshape(-1, 1)
            arr = np.concatenate((arr, np.arange(arr.shape[0]).reshape(-1, 1)), axis=1)
        return arr

    def _pairwise_dist(self, y: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        # wrapper for scipy's cdist function
        # y should give indexes
        y_vec = self.embedding[y[:, 0].astype(np.int)]
        dist_mat = distance.cdist(self.embedding, y_vec, metric=metric)
        return dist_mat

    def _embed(self, X: np.ndarray):
        if not self.multigraph:
            if not np.issubdtype(X.dtype, np.number):
                raise TypeError("Adjacency matrix should have numeric type")
            if np.ndim(X) != 2:
                raise IndexError("Argument must have dim 2")
            if X.shape[0] != X.shape[1]:
                raise IndexError("Adjacency Matrix should be square.")
        else:
            raise NotImplementedError("Multigraph SVN not implemented")

        # Embed graph if embedding not provided
        if self.embedding is None:
            if issubclass(type(self.embeder), BaseSpectralEmbed):
                self.embedding = self.embeder.fit_transform(X)
            else:
                raise TypeError("No embeder available")

    def _predict(self, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Nominate vertex based on attribute specific distance from the k nearest neighbors.

        Parameters
        ----------
        k : int
            Number of neighbors to consider in nearest neighbors distance weighting.

        Returns
        -------
        An tuple of two np.ndarrays, each with shape = ``(number_vertices, number_attributes_in_seed)``.
        The array at index 0 is the nomination list. For each attribute column, the rows contain indices
        of vertices in original adjacency matrix ordered by likelihood of matching that attribute.
        The array at index 1 is the computed distances, where each element at (i, j) represents the
        distance metric value between vertex i and attribute j.

        """
        num_unique = self.unique_att.shape[0]
        ordered = self.distance_matrix.argsort(axis=1)
        sorted_dists = self.distance_matrix[np.arange(ordered.shape[0]), ordered.T].T
        nd_buffer = np.tile(
            self.attr_labels[ordered[:, :k]], (num_unique, 1, 1)
        ).astype(np.float64)

        # comparison taking place in 3-dim view, coordinates produced are therefore 3D
        inds = np.argwhere(nd_buffer == self.unique_att[:, np.newaxis, np.newaxis])

        # nans are a neat way to operate on attributes individually
        nd_buffer[:] = np.NaN
        nd_buffer[inds[:, 0], inds[:, 1], inds[:, 2]] = sorted_dists[
            inds[:, 1], inds[:, 2]
        ]

        # weighting function. Outer inverse for consistency, makes equivalent to simple
        # ranking by distance in unattributed case, and makes higher ranked vertices
        # naturally have lower distance metric value.
        pred_weights = np.power(np.nansum(np.power(nd_buffer, -1), axis=2), -1).T

        nan_inds = np.argwhere(np.isnan(pred_weights))
        pred_weights[nan_inds[:, 0], nan_inds[:, 1]] = np.inf
        vert_order = np.argsort(pred_weights, axis=0)

        inds = np.tile(self.unique_att, (1, vert_order.shape[0])).T
        inds = np.concatenate((vert_order.reshape(-1, 1), inds), axis=1)
        pred_weights = pred_weights[inds[:, 0], inds[:, 1]]
        return vert_order, pred_weights.reshape(vert_order.shape)

    def predict(self, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Nominate vertex based on distance from the k nearest neighbors of each class,
        or if seed is unattributed, nominates vertices for each seed vertex.
        Wrapper for private method _predict. Methodology is distance based
        ranking.

        Parameters
        ----------
        k : int, default = None
            Number of neighbors to consider if seed is attributed. Defaults to the size of the seed, i.e. all
            seed vertices are considered. Is ignored in the unattributed case, since it only is reasonable to
            consider all vertices.

        Returns
        -------
        Nomination List : np.ndarray
                        shape is ``(number_vertices, number_attributes_in_seed)`` if attributed, or
                        shape is ``(number_vertices, number_vertices_in_seed)`` if unattributed. Each
                        column is an attribute or seed vertex, and the rows of each column are a list of
                        vertex indexes from the original adjacency matrix in order degree of match.
        Distance Matrix : np.ndarray
                        The matrix of distances associated with each element of the nomination list.
        """
        if k is not None and type(k) is not int:
            raise TypeError("k must be an integer")
        elif k is not None and k <= 0:
            raise ValueError("k must be greater than 0")
        if self.unique_att.shape[0] == self.attr_labels.shape[0] or k is None:
            # seed is not attributed, or no k is specified.
            return self._predict(k=self.unique_att.shape[0])
        else:
            # seed is attributed and k is specified/
            return self._predict(k=k)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Constructs the embedding if not provided, then calculates the pairwise distance from each
        seed to each vertex in graph.

        Parameters
        ----------
        X : np.ndarray
            Adjacency matrix representation of graph. May be None if embedding was provided.
        y: np.ndarray
            List of seed vertex indices, OR List of tuples of seed vertex indices and associated attributes.

        Returns
        -------
        None
        """
        # ensure y has correct shape. If unattributed (1d)
        # add unique attribute to each seed vertex.
        y = self._make_2d(y)
        if not self.persistent or self.embedding is None:
            if X is None:
                raise ValueError(
                    "Adjacency matrix must be provided if embedding is None."
                )
            X = np.array(X)
            self._embed(X)

        self.attr_labels = y[:, 1]
        self.unique_att = np.unique(self.attr_labels)
        self.distance_matrix = self._pairwise_dist(y)

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calls this class' fit and then predict methods.

        Parameters
        ----------
        X : np.ndarray.
            Adjacency matrix representation of graph. May be None if embedding was provided.
        y : np.ndarray.
            List of seed vertex indices in adjacency matrix in column 1, and associated attributes in column 2,
            OR list of unattributed vertex indices.
        k : int, default = None
            Number of neighbors to consider if seed is attributed. Defaults to the size of the seed, i.e. all
            seed vertices are considered. Is ignored in the unattributed case, since it only is reasonable to
            consider all vertices.

        Returns
        -------
        Nomination List : np.ndarray
                        shape is ``(number_vertices, number_attributes_in_seed)`` if attributed, or
                        shape is ``(number_vertices, number_vertices_in_seed)`` if unattributed. Each
                        column is an attribute or seed vertex, and the rows of each column are a list of
                        vertex indexes from the original adjacency matrix in order degree of match.
        Distance Matrix : np.ndarray
                        The matrix of distances associated with each element of the nomination list.
        """
        self.fit(X, y)
        return self.predict(k=k)
