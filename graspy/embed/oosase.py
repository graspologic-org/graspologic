# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import networkx as nx
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from graspy.utils import is_fully_connected

from .base import BaseEmbed


class OOSAdjacencySpectralEmbed(BaseEmbed):
    r"""
    Class for computing the out of sample adjacency spectral embedding of a graph.
    
    The adjacency spectral embedding (ASE) is a k-dimensional Euclidean representation 
    of the graph based on its adjacency matrix [1]_. It relies on an SVD to reduce the 
    dimensionality to the specified k, or if k is unspecified, can find a number of
    dimensions automatically (see graspy.embed.selectSVD). The out of sample adjacency
    spectral embedding (OOSASE) considers the ASE of an induced subgraph of the original
    graph. To embed "out of sample" vertices, a projection matrix learned from the in
    sample embedding is used [2]_.

    Parameters
    ----------
    n_components : int or None, default = None
        Desired dimensionality of output data. If "full", 
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        ``select_dimension`` using ``n_elbows`` argument.
    n_elbows : int, optional, default: 2
        If `n_components=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.
    algorithm : {'randomized' (default), 'full', 'truncated'}, optional
        SVD solver to use:

        - 'randomized'
            Computes randomized svd using 
            ``sklearn.utils.extmath.randomized_svd``
        - 'full'
            Computes full svd using ``scipy.linalg.svd``
        - 'truncated'
            Computes truncated svd using ``scipy.sparse.linalg.svd``
    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or 
        'truncated'. The default is larger than the default in randomized_svd 
        to handle sparse matrices that may have large slowly decaying spectrum.
    check_lcc : bool , optional (default = True)
        Whether to check if input graph is connected. May result in non-optimal 
        results if the graph is unconnected. If True and input is unconnected,
        a UserWarning is thrown. Not checking for connectedness may result in 
        faster computation.
    in_sample_proportion : float , optional (default = 1)
        If in_sample_idx is None, the proportion of in sample vertices to use for
        initial embedding.
    in_sample_idx : array-like , optional (default = None)
    connected_attempts : integer , optional (default = 100)
        Number of sets of indices generated to find a connected induced subgraph.
        If the induced subgraph is not connected after connected_attempts then
        a warning is thrown.
    semi_supervised : boolean , optional (default = False)
        If true, once an out-of-sample vertex is embedded it is considered to be
        in-sample, and its' embeddding used for subsequent out-of-sample vertices.
    random_state : integer , optional (default = None)
        Random seed used to generate in sample indices.

    Attributes
    ----------
    singular_values_ : array, shape (n_components)
        Singular values associated with the latent position matrices
    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph. 
    latent_right_ : array, shape (n_samples, n_components), or None
        Only computed when the graph is directed, or adjacency matrix is assymetric.
        Estimated right latent positions of the graph. Otherwise, None.
    directed : bool
        True if the graph is directed
    id_to_index : dict
        Mapping from node id to node index (i.e. 'node_id0' -> 0)
    index_to_id : dict
        Mapping from node index to node id (i.e. 0 -> 'node_id0')



    See Also
    --------
    graspy.embed.selectSVD
    graspy.embed.select_dimension

    Notes
    -----
    The singular value decomposition: 

    .. math:: A = U \Sigma V^T

    is used to find an orthonormal basis for a matrix, which in our case is the
    adjacency matrix of the graph. These basis vectors (in the matrices U or V) are
    ordered according to the amount of variance they explain in the original matrix.
    By selecting a subset of these basis vectors (through our choice of dimensionality
    reduction) we can find a lower dimensional space in which to represent the graph.

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    .. [2] Levin, K., Roosta-Khorasani, F., Mahoney, M. W., & Priebe, C. E. (2018).
        Out-of-sample extension of graph adjacency spectral embedding. PMLR: Proceedings
        of Machine Learning Research, 80, 2975-2984.
    """

    def __init__(
        self,
        n_components=None,
        n_elbows=2,
        algorithm="randomized",
        n_iter=5,
        check_lcc=True,
        in_sample_proportion=1,
        in_sample_idx=None,
        connected_attempts=100,
        semi_supervised=False,
        random_state=None,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
            check_lcc=check_lcc,
        )

        if random_state is not None:
            random_state = np.random.randint(random_state)
    
        if in_sample_proportion <= 0 or in_sample_proportion > 1:
            if in_sample_idx is None:
                msg = (
                    "must give either proportion of in sample indices or a list"
                    + " of in sample vertices"
                )
                raise ValueError(msg)
        self.connected_attempts = connected_attempts
        self.semi_supervised = semi_supervised
        self.in_sample_proportion = in_sample_proportion
        self.in_sample_idx = in_sample_idx

    def fit(self, graph, y=None):
        """
        Fit ASE model to input graph

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to embed.

        Returns
        -------
        self : returns an instance of self.
        """

        if isinstance(graph, np.ndarray):
            check_array(graph)
            graph = nx.Graph(graph).copy()

        if isinstance(graph, nx.Graph):
            if nx.is_directed(graph):
                self.directed = True
            else:
                self.directed = False
        else:
            msg = "only arrays and networkx graphs allowed"
            raise TypeError(msg)

        self._nodes = list(graph.nodes)

        if self.check_lcc:
            if not is_fully_connected(graph):
                msg = (
                    "Input graph is not fully connected. Results may not"
                    + " be optimal. You can compute the largest connected component by"
                    + " using ``graspy.utils.get_lcc``."
                )
                warnings.warn(msg, UserWarning)

        n_verts = len(graph)

        if self.in_sample_proportion is None:
            self.in_sample_proportion = len(self.in_sample_idx) / n_verts

        if self.in_sample_idx is None:
            if self.in_sample_proportion == 1:
                self.in_sample_idx = self._nodes
            else:
                self.in_sample_idx = np.random.choice(
                    self._nodes, int(n_verts * self.in_sample_proportion)
                )
        else:
            self.in_sample_proportion = len(self.in_sample_idx) / n_verts

        self._connected_subgraph(graph)

        in_sample_G = graph.subgraph(self.in_sample_idx).copy()

        in_sample_A = nx.to_numpy_array(in_sample_G)

        self.id_to_index = dict(zip(self.in_sample_idx, range(len(self.in_sample_idx))))
        self.index_to_id = dict(zip(range(len(self.in_sample_idx)), self.in_sample_idx))

        self._reduce_dim(in_sample_A)

        return self

    def _connected_subgraph(self, graph):
        """
        Find a connected subgraph.

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to embed.

        Returns
        -------
        None

        """
        n_verts = len(graph)
        in_sample_G = graph.subgraph(self.in_sample_idx).copy()

        if is_fully_connected(in_sample_G):
            return

        if self.in_sample_proportion < 1:
            c = 0
            less_than_half = True
            while (
                not (is_fully_connected(in_sample_G) and c < self.connected_attempts)
                and less_than_half
            ):
                self.in_sample_idx = np.random.choice(
                    self._nodes, int(n_verts * self.in_sample_proportion)
                )
                temp = graph.subgraph(self.in_sample_idx).copy()
                temp_lcc = max(nx.connected_components(temp), key=len)
                if len(temp_lcc) / n_verts > self.in_sample_proportion / 2:
                    self.in_sample_idx = temp_lcc
                    less_than_half = False
                c += 1
            if c == self.connected_attempts:
                msg = (
                    "Induced subgraph is not fully connected."
                    + "Attempted to find connected"
                    + " induced subgraph {} times. Results may not be optimal.".format(
                        self.connected_attempts
                    )
                    + " Try increasing proportion of in sample vertices."
                )
                warnings.warn(msg, UserWarning)

        return

    def predict(self, X, ids=None):
        """
        Embed out of sample vertices.

        Parameters
        ----------
        X : array_like, shape (m, n)
            m stacked similarity lists, where the jth entry of the ith row corresponds
            to the similarity of the ith out of sample observation to the jth in sample
            observation.
        ids : array_like, length m
            If self.semi_supervised the list must be of length m and contain the node 
            ids for the nodes to embed. Otherwise, it is a no-op.

        Returns
        -------
        oos_embedding : array, shape (m, d)
            The embedding of the m out of sample vertices.
        """

        # Check if fit is already called
        check_is_fitted(self, ["latent_left_"], all_or_any=all)

        if self.directed:
            in_sample_embedding = np.concatenate(
                (self.latent_left_, self.latent_right_), axis=1
            )
        else:
            in_sample_embedding = self.latent_left_

        n_in_sample_verts, d = in_sample_embedding.shape

        # Type checking
        check_array(
            X,
            ensure_2d=False,
            allow_nd=False,
            ensure_min_samples=1,
            ensure_min_features=n_in_sample_verts,
        )

        if X.ndim == 1:
            X = X.reshape((1, -1))
            m = 1
        elif X.shape[1] != n_in_sample_verts:
            msg = "Similarity vector must be of length n"
            raise ValueError(msg)
        else:
            m = X.shape[0]

        row_sums = np.sum(X, axis=1)
        if np.count_nonzero(row_sums) != m:
            msg = (
                "At least one adjacency vector is the zero vector."
                + " It is recommended to first embed nodes with non-zero adjacency"
                + " vectors with self.semi_supervised = True and embed the nodes"
                + " with zero adjacencies"
            )
            warnings.warn(msg, UserWarning)

        oos_embedding = X @ np.linalg.pinv(in_sample_embedding).T

        if self.semi_supervised:
            if ids is None:
                if set(self.in_sample_idx) == set(range(n_in_sample_verts)):
                    ids = np.arange(n_in_sample_verts, n_in_sample_verts + m)
                else:
                    msg = "Semi supervised embedding without node ids."
                    raise ValueError(msg)
            elif len(ids) != m:
                msg = (
                    "Length of node ids,"
                    + " {}, does not match number of out of sample nodes, {}".format(
                        len(ids), m
                    )
                )
                raise ValueError(msg)

            if self.directed:
                self.latent_left_ = np.concatenate(
                    (self.latent_left_, oos_embedding[:, : int(d / 2)])
                )
                self.right_left_ = np.concatenate(
                    (self.right_left_, oos_embedding[:, int(d / 2) : d])
                )
            else:
                self.latent_left_ = np.concatenate((self.latent_left_, oos_embedding))

            self.id_to_index.update(
                dict(zip(range(n_in_sample_verts, n_in_sample_verts + m), ids))
            )
            self.index_to_id.update(
                dict(zip(ids, range(n_in_sample_verts, n_in_sample_verts + m)))
            )

        return oos_embedding

    def fit_predict(self, graph, edge_weight_attr=None):
        """
        Perform both in sample and out of sample adjacency spectral embedding. If
        the graph is too big, this function will break.

        Only unweighted graphs supported.

        Parameters
        ----------
        graph : array-like or networkx.Graph
            Input graph to embed.
        edge_weight_attr : string
            The edge weight attribute. If None, it is assumed the graph is binary.


        Returns
        -------
        embedding : array
            Embedding of in sample and out of sample vertices.
        """

        self.fit(graph)

        n_verts = len(graph)

        if self.directed:
            in_sample_embedding = np.concatenate(
                (self.latent_left_, self.latent_right_), axis=1
            )
        else:
            in_sample_embedding = self.latent_left_

        n_in_sample_verts, d = in_sample_embedding.shape

        in_sample_names = [self.index_to_id[i] for i in self.in_sample_idx]
        out_sample_names = [i for i in self._nodes if i not in in_sample_names]

        if isinstance(graph, nx.Graph):
            out_sample_edges = [graph.edges([os_id]) for os_id in out_sample_names]

            X = np.zeros((n_verts - n_in_sample_verts, n_in_sample_verts))
            for i in range(len(out_sample_id)):
                for edge in out_sample_edges[i]:
                    if edge[1] in self.in_sample_idx:
                        if edge_weight_attr is None:
                            X[i, self._node_id_map[edge[1]]] += 1
                        else:
                            # this might be broken
                            X[i, self._node_id_map[edge[1]]] += graph.get_edge_data(
                                out_sample_names[i], edge[1], default=0
                            )[edge_weight_attr]

        out_sample_idx = np.array(
            [i for i in range(n_verts) if i not in self.in_sample_idx]
        )
        if isinstance(graph, np.ndarray):
            check_array(graph)
            X = graph[np.ix_(out_sample_idx, self.in_sample_idx)]

        # this might be broken
        oos = self.predict(X, ids=out_sample_names)

        all_nodes_in_order = np.concatenate((self.in_sample_idx, out_sample_names))

        embedding = np.zeros((n_verts, in_sample_embedding.shape[1]))
        embedding[:n_in_sample_verts] = in_sample_embedding[:n_in_sample_verts]
        embedding[out_sample_idx] = oos

        self.index_to_id = dict(zip(range(n_verts), all_nodes_in_order))
        self.id_to_index = dict(zip(all_nodes_in_order, range(n_verts)))

        if self.semi_supervised:
            self._nodes = all_nodes_in_order

            if self.directed:
                self.latent_left_ = embedding[:, : int(d / 2)]
                self.latent_right_ = embedding[:, int(d / 2) : d]
            else:
                self.latent_left_ = embedding

        return embedding
