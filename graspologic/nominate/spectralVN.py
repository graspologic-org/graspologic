# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Union, Tuple
from ..embed import BaseSpectralEmbed
from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator


class SpectralVertexNomination(BaseEstimator):
    """
    Class for spectral vertex nomination on a single graph.

    Given a graph :math:`G=(V,E)` and a subset of :math:`V` called :math:`S`
    (the "seed"), Single Graph Vertex Nomination is the problem of ranking all :math:`V`
    in order of relation to members of :math:`S`. Spectral Vertex Nomination solves
    this problem by embedding :math:`G` into a low dimensional euclidean space
    (see: `Adjacency Spectral Embed Tutorial
    <https://microsoft.github.io/graspologic/tutorials/embedding/AdjacencySpectralEmbed.html>`_
    ), and then generating a nomination list by some distance based algorithm. In the
    simple unattributed case, for each seed vertex :math:`u`, the other vertices are
    ranked in order of euclidean distance from :math:`u`.

    Parameters
    ----------
    input_graph: bool, default = True
        Flag whether to expect two full graphs, or the embeddings.

        - True
            .fit and .fit_predict() expect graphs as adjacency matrix, provided as
            ndarray of shape (n, n). They will be embedded using the specified
            ``embedder``.
        - False
            .fit() and .fit_predict() expect an embedding of the graph, i.e. a ndarray
            of size (n, d).
    embedder: str or BaseEmbed, default = 'ASE'
        May provide either a embed object or a string indicating which embedding method
        to use, which may be either:
        "ASE" for :py:class:`~graspologic.embed.AdjacencySpectralEmbed` or
        "LSE" for :py:class:`~graspologic.embed.LaplacianSpectralEmbed`.
    n_neighbors: int, default=None
        The number of vertices to nominate for each seed.
    metric : str, default = 'euclidean'
        Distance metric to use when finding the nearest neighbors, all sklearn metrics
        available.
    metric_params : dict, default = None
        Arguments for the sklearn `DistanceMetric` specified via `metric` parameter.

    Attributes
    ----------
    nearest_neighbors_ : sklearn.neighbors.NearestNeighbors
        A fit sklearn `NearestNeighbors` classifier used to find closest vertices to
        each seed.

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
        self,
        input_graph: bool = True,
        embedder: Union[str, BaseSpectralEmbed] = "ase",
        n_neighbors=None,
        metric: str = "euclidean",
        metric_params: dict = None,
    ):
        super().__init__()
        self.embedder = embedder
        self.input_graph = input_graph
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = metric_params
        self._check_params()

    def _check_x(self, X: np.ndarray):
        # check X
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be of type np.ndarray.")
        elif not np.issubdtype(X.dtype, np.number):
            raise TypeError("Adjacency matrix or embedding should have numeric type")
        elif np.ndim(X) != 2:
            raise IndexError("Adjacency matrix or embedding must have dim 2")
        elif not self.input_graph:
            # embedding was provided
            if X.shape[1] > X.shape[0]:
                raise IndexError("Dim 1 of an embedding should be smaller than dim 0.")
            if not np.issubdtype(X.dtype, np.float):
                raise TypeError("Embedding should have type float")
        elif not np.issubdtype(X.dtype, np.int):
            raise TypeError("Adjacency matrix should have type int")
        elif X.shape[0] != X.shape[1]:
            raise IndexError("Adjacency Matrix should be square.")

    def _check_y(self, y: np.ndarray):
        # check y
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be of type np.ndarray")
        elif not np.issubdtype(y.dtype, np.integer):
            raise TypeError("y must have dtype int")
        elif np.ndim(y) > 2 or (y.ndim == 2 and y.shape[1] > 1):
            raise IndexError("y must have shape (n) or (n, 1).")
        elif y.shape[0] > self.embedding_.shape[0]:
            raise ValueError(
                "the number of seeds must be less than the number of vertices"
            )

    def _check_params(self):

        if self.n_neighbors is not None and type(self.n_neighbors) is not int:
            raise TypeError("k must be an integer")
        elif self.n_neighbors is not None and self.n_neighbors <= 0:
            raise ValueError("k must be greater than 0")
        if type(self.metric) is not str:
            raise TypeError("metric must be a string")
        if self.metric_params is not None and type(self.metric_params) is not dict:
            raise TypeError("metric_params must be a dictionary")
        if type(self.input_graph) is not bool:
            raise TypeError("input_graph_ must be of type bool.")
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
            elif not isinstance(self.embedder, str):
                raise TypeError("embedder must be type str or BaseSpectralEmbed")
            elif self.embedder.lower() == "ase":
                embedder = AdjacencySpectralEmbed()
            elif self.embedder.lower() == "lse":
                embedder = LaplacianSpectralEmbed()
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
        y=None,
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
        y : NoneType
            Included by sklearn convention.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # ensure y has correct shape. If unattributed (1d)
        # add unique attribute to each seed vertex.
        self._check_x(X)
        self._embed(X)
        if self.n_neighbors is None:
            self.n_neighbors = X.shape[0]
        self.nearest_neighbors_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            metric_params=self.metric_params,
        )
        self.nearest_neighbors_.fit(self.embedding_)
        return self

    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Nominates vertices for each seed vertex. Methodology is distance based ranking.


        Parameters
        ----------
        y : np.ndarray
          The indices of the seed vertices. Should be a dim 1 array with length less
          than :math:`|V|`.

        Returns
        -------
        Nomination List : np.ndarray
                        Shape is ``(number_vertices, number_vertices_in_seed)`` . Each
                        column is a seed vertex, and the rows of each
                        column are a list of vertex indexes from the original adjacency
                        matrix in order degree of match.
        Distance Matrix : np.ndarray
                        The matrix of distances associated with each element of the
                        nomination list.
        """
        self._check_y(y)
        y = y.reshape(-1)
        y_vec = self.embedding_[y.astype(np.int)]
        if not hasattr(self, "nearest_neighbors_"):
            raise ValueError("Fit must be called before predict.")
        distance_matrix, nomination_list = self.nearest_neighbors_.kneighbors(y_vec)
        # transpose for consistency with literature
        return nomination_list.T.astype(np.int), distance_matrix.T

    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
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
            List of unattributed seed vertex indices.

        Returns
        -------
        Nomination List : np.ndarray
                        Shape is ``(number_vertices, number_vertices_in_seed)`` . Each
                        column is a seed vertex, and the rows of each
                        column are a list of vertex indexes from the original adjacency
                        matrix in order degree of match.
        Distance Matrix : np.ndarray
                        The matrix of distances associated with each element of the
                        nomination list.
        """
        self.fit(X)
        return self.predict(y)
