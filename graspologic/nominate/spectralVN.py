# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Union, Tuple
from .base import BaseVN
from ..embed import BaseSpectralEmbed
from ..embed import AdjacencySpectralEmbed as ase, LaplacianSpectralEmbed as lse
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SpectralVertexNomination(BaseVN):
    """
    Class for spectral vertex nomination on a single graph.

    Given a graph :math:`G=(V,E)` and a subset of :math:`V` called :math:`S`
    (the "seed"), Single Graph Vertex Nomination is the problem of ranking all :math:`V`
    in order of relation to members of :math:`S`. Spectral Vertex Nomination solves
    this problem by embedding :math:`G` into a low dimensional euclidean space
    (:ref:`tutorials <embed_tutorials>`), and then generating a nomination list by some
    distance based algorithm. In the simple unattributed case, for each seed vertex
    :math:`u`, the other vertices are ranked in order of euclidean distance from
    :math:`u`. In the attributed case, vertices are ranked by relatedness to each
    attribute present in the set of seed vertices.

    Parameters
    ----------
    input_graph: bool, default = True
        Flag whether to expect two full graphs, or the embeddings.

        - True
            .fit and .fit_predict() expect graphs as adjacency matrix, provided as
            ndarray of shape (n, n). They will be embedded using the specified embedder.
        - False
            .fit() and .fit_predict() expect an embedding of the graph, i.e. a ndarray
            of size (n, d).
    embedder: str or BaseEmbed, default = 'ASE'
        May provide either a embed object or a string indicating which embedding method
        to use, which may be either:
        "ASE" for :py:class:`~graspologic.embed.AdjacencySpectralEmbed` or
        "LSE" for :py:class:`~graspologic.embed.LaplacianSpectralEmbed`.

    Attributes
    ----------
    attribute_labels_ : np.ndarray
        The attributes of the vertices in the seed (parameter 'y' for fit). Shape is the
        number of seed vertices. Each value is unique in the unattributed case.
    unique_attributes_ : np.ndarray
        Each unique attribute represented in the seed. One dimensional. In the
        unattributed case of SVN, the number of unique attributes (and therefore the
        shape along axis 0 of `unique_att_`) is equal to the number of seeds ( the shape
        along axis 0 of `attribute_labels_`).
    distance_matrix_ : np.ndarray
        The euclidean distance from each seed vertex to each vertex. Shape is
        ``(number_vertices, k)`` if attributed or shape is
        ``(number_vertices, number_seed_vertices)`` if unattributed.

    References
    ----------
    .. [1] Fishkind, D. E.; Lyzinski, V.; Pao, H.; Chen, L.; Priebe, C. E. Vertex
        nomination schemes for membership prediction. Ann. Appl. Stat. 9 2015.
        https://projecteuclid.org/euclid.aoas/1446488749

    .. [2] Jordan Yoder, Li Chen, Henry Pao, Eric Bridgeford, Keith Levin,
        Donniell E. Fishkind, Carey Priebe, Vince Lyzinski, Vertex nomination: The
        canonical sampling and the extended spectral nomination schemes, Computational
        Statistics & Data Analysis, Volume 145, 2020.
        http://www.sciencedirect.com/science/article/pii/S0167947320300074

    """

    def __init__(
        self, input_graph: bool = True, embedder: Union[str, BaseSpectralEmbed] = "ASE"
    ):
        super().__init__(multigraph=False)
        self.embedder = embedder
        self.input_graph = input_graph
        self.neighbor_inds_ = None
        self.attribute_labels_ = None
        self.unique_attributes_ = None
        self.distance_matrix_ = None
        self.embedding_ = None

    @staticmethod
    def _make_2d(arr: np.ndarray) -> np.ndarray:
        # ensures arr is two or less dimensions.
        # if 1d, adds unique at each index on
        # the second dimension.
        if np.ndim(arr) == 1 or arr.shape[1] == 1:
            arr = arr.reshape(-1, 1)
            arr = np.concatenate((arr, np.arange(arr.shape[0]).reshape(-1, 1)), axis=1)
        return arr

    def _check_inputs(self, X: np.ndarray, y: np.ndarray, k: int):
        if type(self.input_graph) is not bool:
            raise TypeError("input_graph_ must be of type bool.")
        # check X
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be of type np.ndarray.")
        if not self.multigraph:
            if not np.issubdtype(X.dtype, np.number):
                raise TypeError(
                    "Adjacency matrix or embedding should have numeric type"
                )
            elif np.ndim(X) != 2:
                raise IndexError("Adjacency matrix or embedding must have dim 2")
            elif not self.input_graph:
                # embedding was provided
                if X.shape[1] > X.shape[0]:
                    raise IndexError(
                        "dim 1 of an embedding should be smaller than dim 0."
                    )
            elif X.shape[0] != X.shape[1]:
                raise IndexError("Adjacency Matrix should be square.")
        else:
            raise NotImplementedError("Multigraph SVN not implemented")
        # check y
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be of type np.ndarray")
        elif not np.issubdtype(y.dtype, np.integer):
            raise TypeError("y must have dtype int")
        elif np.ndim(y) > 2 or (y.ndim == 2 and y.shape[1] > 2):
            raise IndexError("y must have shape (n) or (n, 1) or (n, 2).")
        # check k
        if k is not None and type(k) is not int:
            raise TypeError("k must be an integer")
        elif k is not None and k <= 0:
            raise ValueError("k must be greater than 0")

        if self.input_graph:
            if not isinstance(self.embedder, BaseSpectralEmbed) and not isinstance(
                self.embedder, str
            ):
                raise TypeError(
                    "embedder must be either of type str or BaseSpectralEmbed"
                )

    def _embed(self, X: np.ndarray):
        # Embed graph if embedding not provided
        if self.input_graph:
            if isinstance(self.embedder, BaseSpectralEmbed):
                embedder = self.embedder
            elif self.embedder == "ASE":
                embedder = ase()
            elif self.embedder == "LSE":
                embedder = lse()
            else:
                raise ValueError(
                    "Requested embedding method does not exist, if str is passed must "
                    "be either 'ASE' or 'LSE'."
                )
            self.embedding_ = embedder.fit_transform(X)
        else:
            self.embedding_ = X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = None,
        metric: str = "euclidean",
        metric_params: dict = None,
    ):
        """
        Constructs the embedding if not provided, then calculates the pairwise distance
        from each seed to each vertex in graph.

        Parameters
        ----------
        X : np.ndarray
            - If `input_graph` is True
                Expects a graph as an adjacency matrix, i.e. an ndarray of shape (n, n).
                Will be embedded using the specified embedder.
            - If `input_graph` is False
                Expects an embedding of the graph, i.e. a ndarray of size (n, d).
        y: np.ndarray
            Shape (n) array of seed vertex indices, OR shape (2, n) array of seed
            seed vertex indices and associated attributes.
        k : int, default = None
            Number of neighbors to consider if seed is attributed. Defaults to the size
            of the seed, i.e. all seed vertices are considered. Is ignored in the
            unattributed case, since it only is reasonable to consider all vertices.
        metric : str, default = 'euclidean'
            Distance metric to use in computing nearest neighbors, all sklearn metrics
            available.
        metric_params : dict, default = None
            Arguments for the sklearn `DistanceMetric` specified via `metric` parameter.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # ensure y has correct shape. If unattributed (1d)
        # add unique attribute to each seed vertex.
        self._check_inputs(X, y, k)
        y = self._make_2d(y)
        self._embed(X)

        self.attribute_labels_ = y[:, 1]
        self.unique_attributes_ = np.unique(self.attribute_labels_)
        if (
            self.unique_attributes_.shape[0] == self.attribute_labels_.shape[0]
            or k is None
        ):
            # seed is not attributed, or no k is specified.
            k = self.attribute_labels_.shape[0]

        nearest_neighbors = NearestNeighbors(
            n_neighbors=k, metric=metric, metric_params=metric_params
        )
        y_vec = self.embedding_[y[:, 0].astype(np.int)]
        nearest_neighbors.fit(y_vec)
        self.distance_matrix_, self.neighbor_inds_ = nearest_neighbors.kneighbors(
            self.embedding_, return_distance=True
        )
        return self

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Nominate vertex based on distance from the k nearest neighbors of each class,
        or if seed is unattributed, nominates vertices for each seed vertex.
        Methodology is distance based ranking.

        Returns
        -------
        Nomination List : np.ndarray
                        Shape is ``(number_vertices, number_attributes_in_seed)`` if
                        attributed, or shape is
                        ``(number_vertices, number_vertices_in_seed)`` if unattributed.
                        Each column is an attribute or seed vertex, and the rows of each
                        column are a list of vertex indexes from the original adjacency
                        matrix in order degree of match.
        Distance Matrix : np.ndarray
                        The matrix of distances associated with each element of the
                        nomination list.
        """
        k = self.neighbor_inds_.shape[1]
        num_unique = self.unique_attributes_.shape[0]

        # nearest neighbors orders seed for each vertex, we want to order the vertices
        # for each seed or attribute. The following code does that in an efficient
        # manner.

        # nd_buffer provides a single k x n x m (m is number of unique attributes) view
        # for doing efficient high dimensional operations.

        # first we find the indexes of attributes separated across first m dimensions.
        nd_buffer = np.tile(
            self.attribute_labels_[self.neighbor_inds_[:, :k]], (num_unique, 1, 1)
        ).astype(np.float64)
        inds = np.argwhere(
            nd_buffer == self.unique_attributes_[:, np.newaxis, np.newaxis]
        )

        # nd_buffer filled with nan, then distances across the attribute dimensions.
        nd_buffer[:] = np.NaN
        nd_buffer[inds[:, 0], inds[:, 1], inds[:, 2]] = self.distance_matrix_[
            inds[:, 1], inds[:, 2]
        ]

        # weighting function. Conditional is not needed, but avoids unnecessary
        # computation in the unattributed case.
        if num_unique != len(self.attribute_labels_):
            # is attributed. Distances corresponding to seeds of the same attribute
            # are combined via sum of inverse distance, so we get one distance per
            # attribute. Outer inverse is for consistency with unattributed.
            pred_weights = np.power(np.nansum(np.power(nd_buffer, -1), axis=2), -1).T
        else:
            # nansum to collapse nd_buffer, will only be one non-nan across
            # attribute dimension (dim 2).
            pred_weights = np.nansum(nd_buffer, axis=2).T

        # make sure any nans are given infinite weight.
        nan_inds = np.argwhere(np.isnan(pred_weights))
        pred_weights[nan_inds[:, 0], nan_inds[:, 1]] = np.inf

        # sort all vertices for each seed / attribute.
        vert_order = np.argsort(pred_weights, axis=0)

        # batch compute the 2D indices matrix for sorting distances.
        inds = np.tile(self.unique_attributes_, (1, vert_order.shape[0])).T
        inds = np.concatenate((vert_order.reshape(-1, 1), inds), axis=1)

        # produce sorted distances.
        pred_weights = pred_weights[inds[:, 0], inds[:, 1]]

        # return without empty dimensions.
        return vert_order, pred_weights.reshape(vert_order.shape)

    def fit_predict(
        self, X: np.ndarray, y: np.ndarray, k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calls this class' fit and then predict methods.

        Parameters
        ----------
        X : np.ndarray
            - If `input_graph` is True
                Expects a graph as an adjacency matrix, i.e. an ndarray of shape (n, n).
                Will be embedded using the specified embedder.
            - If `input_graph` is False
                Expects an embedding of the graph, i.e. a ndarray of size (n, d).
        y : np.ndarray.
            List of seed vertex indices in adjacency matrix in column 1, and associated
            attributes in column 2, OR list of unattributed vertex indices.
        k : int, default = None
            Number of neighbors to consider if seed is attributed. Defaults to the size
            of the seed, i.e. all seed vertices are considered. Is ignored in the
            unattributed case, since it only is reasonable to consider all vertices.

        Returns
        -------
        Nomination List : np.ndarray
                        Shape is ``(number_vertices, number_attributes_in_seed)`` if
                        attributed, or shape is
                        ``(number_vertices, number_vertices_in_seed)`` if unattributed.
                        Each column is an attribute or seed vertex, and the rows of each
                        column are a list of vertex indexes from the original adjacency
                        matrix in order degree of match.
        Distance Matrix : np.ndarray
                        The matrix of distances associated with each element of the
                        nomination list.
        """
        self.fit(X, y, k=k)
        return self.predict()
