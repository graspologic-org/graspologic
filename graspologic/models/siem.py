# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Any, Collection, Optional

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

    There is no argument for ``loops`` because the diagonal is assumed
    to be ommitted from the ``edge_clust`` if loops are not relevant. 
    There is no argument for ``directed`` because the lower triangle is
    assumed to be omitted if the network is undirected.
    
    Parameters
    ----------
    directed : bool, default = True
    Whether the network should be interpreted to be directed (or not).
    If the network is taken to be directed, *only* the upper triangle
    of the ``edge_clust`` will be used in computation of the network's
    properties.

    Attributes
    ----------
    model : dict
    
    a dictionary of community names to a dictionary of edge indices and weights.

    K : int 
    the number of unique communities in the network.

    n_vertices : int
    the number of vertices in the graph.

    See also
    --------
    graspologic.simulations.siem
    """

    def __init__(
        self, 
        directed: bool = True
    ):
        super().__init__(directed=directed, loops=loops)
        self.model = {}
        self.K = None
        self._has_been_fit = False

    def fit(
        self,
        graph: GraphRepresentation, 
        edge_clust: array_like,
    ) -> None:
        """
        Fits an SIEM to a graph.

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit

        edge_clust : array_like, shape graph.shape
            A matrix giving the community assignments for each edge within the adjacency matrix
            of `graph`.
        """
        graph = import_graph(graph)
        
        self.n_vertices = graph.shape[0]
        if not np.ndarray.all(np.isfinite(graph)):
            raise ValueError("`graph` has non-finite entries.")
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("`graph` is not a square adjacency matrix.")
        if edge_clust.shape[0] != edge_clust.shape[1]:
            raise ValueError("`edge_clust` is not a square matrix.")
        if not graph.shape == edge_clust.shape:
            msg = """
            Your edge communities do not have the same number of vertices as the graph.
            Graph has %d vertices; edge community has %d vertices.
            """.format(
                graph.shape[0], edge_clust.shape[0]
            )
            raise ValueError(msg)
        bool_mtx = np.ones((n_vertices, n_vertices), dtype=bool)
        if not self.directed:
            bool_mtx[np.tril_indices(n_vertices, k=0)] = False

        self.cluster_names = np.unique(edge_clust)
        siem = {
            x: {"edges": np.where(edge_clust == x & bool_mtx), 
                "weights": graph[edge_clust == x & bool_mtx],
                "prob": np.mean(graph[edge_clust == x] != 0 & bool_mtx)}
            for x in self.cluster_names
        }
        self.model = siem
        self.k_clusters = len(self.model.keys())
        self.clust_p_ = {x: siem[x]["prob"] for x in siem.keys()}
        self.edge_clust = edge_clust
        self.p_mat_ = _clust_to_full(self.edge_clust, self.clust_p)
        if self._has_been_fit:
            warnings.warn("A model has already been fit. Overwriting previous model...")
        self._has_been_fit = True
        return

    def edgeclust_from_commvec(
        self,
        y : array_like,
        loops : bool = False,
    ) -> np.ndarray:
        """
        A function which takes a vector of labels for an SBM and converts it
        to an analogous SIEM edge cluster matrix.

        Parameters
        ----------
        y : array_like, length n_vertices
        the labels vector for the nodes in the graph, with K unique entries.
        
        loops : bool, default=False
        whether the network has loops.

        Returns
        -------
        edge_clusts_ : np.ndarray, shape (n_vertices, n_vertices)
        a matrix which indicates the assigned cluster name for each node
        in the network. The entries will be strings ``(comm1, comm2)``, which are
        from the cartesian product of the unique entries in ``y``.
        """
        edge_clusts_ = np.ndarray((len(y), len(y)), dtype=object)
        edge_clust_names = cartesian_product(y, y)
        for clust in edge_clust_names:
            clustname = "({:s}, {:s})".format(clust[0], clust[1])
            edge_clusts_[y == clust[0], y == clust[1]] = clustname
        if not loops:
            np.diag(edge_clusts_) = np.nan
        return edge_clusts_

    def summarize(
        self, 
        wts : dict, 
        wtargs : dict,
    ) -> dict:
        """
        Allows users to compute summary statistics for each edge community in the model.
        
        Parameters
        ----------
        wts: dict of Callable
            A dictionary of summary statistics to compute for each edge community within the model.
            The keys should be the name of the summary statistic, and each entry should be a callable
            function accepting an unnamed argument for a vector or 1-d array as the first argument.
            Keys are names of the summary statistic, and values are the callable objects themselves.
        wtargs: dict of dictionaries
            A dictionary of dictionaries, where keys correspond to the names of summary statistics,
            and values are dictionaries of the trailing, named, parameters desired for the summary function. The
            keys of `wts` and `wtargs` should be identical.

        Returns
        -------
        summary: dictionary of summary statistics
            A dictionary where keys are edge community names, and values are a dictionary of summary statistics
            associated with each community.
        """
        # check that model has been fit
        if not self._has_been_fit:
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

        summary = {}
        for edge_clust in self.model.keys():
            summary[edge_clust] = {}
            for wt_name, (wt, wtarg) in wt_mod.items():
                # and store the summary statistic for this community
                summary[edge_clust][wt_name] = wt(
                    self.model[edge_clust]["weights"], **wtarg
                )
        return summary

    def compare(
        self, 
        c1 : str, 
        c2 : str, 
        method : Callable = mannwhitneyu, 
        methodargs : dict = None,
    ):
        """
        A function for comparing two edge communities for a difference after a model has been fit.
        
        Parameters
        ----------
        c1: str
            A key in the model, from `self.model.keys()`, to be treated as the first entry
            to the comparison method.
        c2: str
            A key in the model, from `self.model.keys()`, to be treated as the second
            entry to the comparison method.
        method: Callable
            A callable object to use for comparing the two objects. Should accept two unnamed
            leading vectors or 1-d arrays of edge weights.
        methodargs: dict
            A dictionary of named trailing arguments to be passed ot the comparison function of
            interest.

        Returns
        -------
        The comparison method applied to the two communities. The return type depends on the 
        comparison method which is applied.
        """
        if not self._has_been_fit:
            raise UnboundLocalError(
                "You must fit a model with `fit()` before comparing communities in the model."
            )
        if not c1 in self.model.keys():
            raise ValueError("`c1` is not a key for the model.")
        if not c2 in self.model.keys():
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
            self.model[c1]["weights"], self.model[c2]["weights"], **methodargs
        )

def _clust_to_full(
    edge_clust : np.ndarray,
    edge_p_ : dict,
) -> np.ndarray:
    """
    "blows up" a k element dictionary to a probability matrix

    edge_clust : np.ndarray of shape (n_vertices, n_vertices)
    with k_clusters unique entries
    edge_p_ : dict
    with k_clusters keys
    """
    p_mat = np.zeros(edge_clust.shape)
    for x in np.unique(edge_clust):
        p_mat[edge_clust == x] = edge_p_[x]
    return p_mat