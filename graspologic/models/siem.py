# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Any, Collection, Optional
import warnings

import numpy as np
from sklearn.utils import check_X_y

from graspologic.types import Dict, List, Tuple
from collections.abc import Callable

from ..types import GraphRepresentation
from ..utils import (
    augment_diagonal,
    cartesian_product,
    import_graph,
    is_unweighted,
    remove_loops,
    symmetrize,
)
from .base import BaseGraphEstimator, _calculate_p
from scipy.stats import mannwhitneyu


class SIEMEstimator(BaseGraphEstimator):
    """
    Stochastic Independent Edge Model

    Parameters
    ----------
    directed : bool, default = True
    Whether the network should be interpreted to be directed (or not).
    If the network is taken to be directed, *only* the upper triangle
    of the ``edge_clust`` will be used in computation of the network's
    properties.

    loops : bool, default = True
    Whether the network should be interpreted to have loops (or not).
    If the network is taken to be loopless, the diagonal of the
    ``edge_clust`` will be ignored when computing properties about
    the network edge clusters.

    Attributes
    ----------
    model_ : dict
        a dictionary of cluster names to a dictionary of edge indices, weights,
        and probabilities. Edge indices of edges in the cluster are in key ``"edges"``,
        weights are in ``"weights"``, and probabilities are in ``"probs"``.

    p_mat_ : np.ndarray, shape (n_verts, n_verts)
        Probability matrix :math:`P` for the fit model, from which graphs could be
        sampled.

    edge_clust_ : np.ndarray, shape (n_verts, n_verts)
        Edge cluster assignment matrix :math:`T` for the fit model, where entry :math:`T_{ij}`
        indicates the cluster assignment of edge :math:`(i,j)`.

    clust_names_ : list
        The names of each sequential cluster.

    clust_p_ : np.ndarray, shape (k_clusters,)
        The block probability vector :math:`\vec{p}`, where the element :math:`p_k`
        represents the probability of an edge which is assigned to block :math:`k`.

    k_clusters_ : int
        the number of unique clusters in the network.

    has_been_fit_ : bool
        a boolean indicating whether the model has been fit yet. if the model has not been fit,
        many post-hoc summary statistics cannot be run.

    n_verts_ : int
    the number of vertices in the graph.

    See also
    --------
    graspologic.simulations.siem
    """

    def __init__(
        self,
        directed: bool = True,
        loops: bool = False,
    ):
        super().__init__(directed=directed, loops=loops)
        self._has_been_fit_ = False  # type: bool

    def fit(
        self,
        graph: GraphRepresentation,
        y: Optional[Any] = None,
    ) -> BaseGraphEstimator:
        """
        Fits an SIEM to a graph.

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit

        y : array_like, shape graph.shape
            A matrix giving the cluster assignments for each edge within the adjacency matrix
            of `graph`. Each entry :math:`y_{ij}` indicates the the edge cluster assignment of
            edge :math:`(i,j)`.
        """
        graph = import_graph(graph)

        self.n_verts_ = graph.shape[0]
        if self._has_been_fit_:
            warnings.warn("A model has already been fit. Overwriting previous model...")
        if y is None:
            raise ValueError(
                "`y` must be supplied. Unsupervised SIEM does not exist (yet!)."
            )
        if not np.ndarray.all(np.isfinite(graph)):
            raise ValueError("`graph` has non-finite entries.")
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("`graph` is not a square adjacency matrix.")
        if y.shape[0] != y.shape[1]:
            raise ValueError("`y` is not a square matrix.")
        if not graph.shape == y.shape:
            msg = """
            Your edge clusters do not have the same number of vertices as the graph.
            Graph has {:d} vertices; edge clusters have {:d} vertices.
            """.format(
                graph.shape[0], y.shape[0]
            )
            raise ValueError(msg)
        bool_mtx = np.ones((self.n_verts_, self.n_verts_), dtype=bool)
        if not self.directed:
            bool_mtx[np.tril_indices(self.n_verts_, k=0)] = False
        if not self.loops:
            np.fill_diagonal(bool_mtx, False)

        self.clust_names_ = np.unique(y[bool_mtx])

        siem = {
            x: {
                "edges": np.where(np.logical_and(y == x, bool_mtx)),
                "weights": graph[np.logical_and(y == x, bool_mtx)],
                "prob": _calculate_p(graph[np.logical_and(y == x, bool_mtx)]),
            }
            for x in self.clust_names_
        }
        self.model_ = siem
        self.k_clusters_ = len(self.model_.keys())
        self.clust_p_ = {x: siem[x]["prob"] for x in siem.keys()}
        self.edge_clust_ = y
        self.p_mat_ = _clust_to_full(self.edge_clust_, self.clust_p_, self.clust_names_)
        self._has_been_fit_ = True
        return self

    def edgeclust_from_commvec(self, y: np.ndarray) -> np.ndarray:
        """
        A function which takes a vector of labels for an SBM and converts it
        to an analogous SIEM edge cluster matrix.

        Parameters
        ----------
        y : array_like, length graph.shape[0], optional
            Categorical labels for the block assignments of the graph, with K
            unique entries.
        loops : bool, default=False
            whether the network has loops.

        Returns
        -------
        edge_clusts_ : np.ndarray, shape (n_vertices, n_vertices)
            a matrix which indicates the assigned cluster name for each node
            in the network. The entries will be strings ``(comm1, comm2)``, which are
            from the cartesian product of the unique entries in ``y``.
        """
        edge_clusts_ = np.ndarray((len(y), len(y)), dtype="<U22")  # type: np.ndarray
        edge_clust_names = cartesian_product(y, y)
        for clust in edge_clust_names:
            clustname = "({:s}, {:s})".format(str(clust[0]), str(clust[1]))
            submtx = np.outer(np.array(y) == clust[0], np.array(y) == clust[1])
            edge_clusts_[submtx] = clustname
        if not self.loops:
            np.fill_diagonal(edge_clusts_, np.nan)
        return edge_clusts_

    def summarize(
        self,
        wts: dict,
        wtargs: dict,
    ) -> Dict[object, object]:
        """
        Allows users to compute summary statistics for each edge cluster in the model.

        Parameters
        ----------
        wts : dict of Callable
            A dictionary of summary statistics to compute for each edge cluster within the model.
            The keys should be the name of the summary statistic, and each entry should be a callable
            function accepting an unnamed argument for a vector or 1-d array as the first argument.
            Keys are names of the summary statistic, and values are the callable objects themselves.
        wtargs : dict of dict
            A dictionary of dictionaries, where keys correspond to the names of summary statistics,
            and values are dictionaries of the trailing, named, parameters desired for the summary function. The
            keys of `wts` and `wtargs` should be identical.

        Returns
        -------
        summary : dict of object
            A dictionary where keys are edge cluster names, and values are a dictionary of summary statistics
            associated with each cluster.
        """
        # check that model has been fit
        if not self._has_been_fit_:
            raise UnboundLocalError(
                "You must fit a model with `fit()` before summarizing the model."
            )
        # check keys for wt and wtargs are same
        if set(wts.keys()) != set(wtargs.keys()):
            raise ValueError("`wts` and `wtargs` should have the same key names.")
        # check wt for callables
        for key, wt in wts.items():
            if not callable(wt):
                raise TypeError("Each value of `wts` should be a callable object.")
        # check whether wtargs is a dictionary of dictionaries with first entry being None in sub-dicts
        for key, wtarg in wtargs.items():
            if not isinstance(wtarg, dict):
                raise TypeError(
                    "Each value of `wtargs` should be a sub-dictionary of class `dict`."
                )

        # equivalent of zipping the two dictionaries together
        wt_mod = {key: (wt, wtargs[key]) for key, wt in wts.items()}

        summary = {}  # type: Dict
        for edge_clust in self.model_.keys():
            summary[edge_clust] = {}
            for wt_name, (wt, wtarg) in wt_mod.items():
                # and store the summary statistic for this cluster
                summary[edge_clust][wt_name] = wt(
                    self.model_[edge_clust]["weights"], **wtarg
                )
        return summary

    def compare(
        self,
        c1: str,
        c2: str,
        method: Callable = mannwhitneyu,
        methodargs: dict = {},
    ) -> object:
        """
        A function for comparing two edge clusters for a difference after a model has been fit.

        Parameters
        ----------
        c1: str
            A key in the model, from `self.model_.keys()`, to be treated as the first entry
            to the comparison method.
        c2: str
            A key in the model, from `self.model_.keys()`, to be treated as the second
            entry to the comparison method.
        method: Callable
            A callable object to use for comparing the two objects. Should accept two unnamed
            leading vectors or 1-d arrays of edge weights.
        methodargs: dict
            A dictionary of named trailing arguments to be passed ot the comparison function of
            interest.

        Returns
        -------
        The comparison method applied to the two clusters. The return type depends on the
        comparison method which is applied.
        """
        if not self._has_been_fit_:
            raise UnboundLocalError(
                "You must fit a model with `fit()` before comparing clusters in the model."
            )
        if not c1 in self.model_.keys():
            raise ValueError("`c1` is not a key for the model.")
        if not c2 in self.model_.keys():
            raise ValueError("`c2` is not a key for the model.")
        if not callable(method):
            raise TypeError("`method` should be a callable object.")
        if not isinstance(methodargs, dict):
            raise TypeError(
                "`methodargs` should be a dictionary of trailing arguments. Got type %s.".format(
                    type(methodargs)
                )
            )
        return method(
            self.model_[c1]["weights"], self.model_[c2]["weights"], **methodargs
        )


def _clust_to_full(
    edge_clust: np.ndarray,
    edge_p_: dict,
    clust_names: list,
) -> np.ndarray:
    """
    "blows up" a k element dictionary to a probability matrix

    edge_clust : np.ndarray of shape (n_vertices, n_vertices)
    with k_clusters unique entries
    edge_p_ : dict
    with k_clusters keys
    """
    p_mat = np.zeros(edge_clust.shape)
    for x in clust_names:
        p_mat[edge_clust == x] = edge_p_[x]
    return p_mat
