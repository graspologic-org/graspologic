import numpy as np

from ..utils import (
    import_graph,
    is_unweighted,
    remove_loops,
    symmetrize,
)
from .base import BaseGraphEstimator, _calculate_p
import warnings


class SIEMEstimator(BaseGraphEstimator):
    """
    Stochastic Independent Edge Model
    
    Parameters
    ----------
    directed : boolean, optional (default=True)
        Whether to treat the input graph as directed. Even if a directed graph is inupt, 
        this determines whether to force symmetry upon the block probability matrix fit
        for the SBM. It will also determine whether graphs sampled from the model are 
        directed. 
    loops : boolean, optional (default=False)
        Whether to allow entries on the diagonal of the adjacency matrix, i.e. loops in 
        the graph where a node connects to itself. 
    Attributes
    ----------
    model: a dictionary of community names to a dictionary of edge indices and weights.
    K: the number of unique edge communities.
    n_vertices: the number of vertices in the graph.
    See also
    --------
    graspy.simulations.siem
    """

    def __init__(self, directed=True, loops=False):
        super().__init__(directed=directed, loops=loops)
        self.model = {}
        self.K = None
        self._has_been_fit = False

    def fit(self, graph, edge_comm):
        """
        Fits an SIEM to a graph.
        Parameters
        ----------
        graph : array_like [nxn] or networkx.Graph with n vertices
            Input graph to fit
        edge_comm : array_like [n x n]
            A matrix giving the community assignments for each edge within the adjacency matrix
            of `graph`.
        """
        graph = import_graph(graph)

        self.n_vertices = graph.shape[0]
        if not np.ndarray.all(np.isfinite(graph)):
            raise ValueError("`graph` has non-finite entries.")
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("`graph` is not a square adjacency matrix.")
        if edge_comm.shape[0] != edge_comm.shape[1]:
            raise ValueError("`edge_comm` is not a square matrix.")
        if not graph.shape == edge_comm.shape:
            msg = """
            Your edge communities do not have the same number of vertices as the graph.
            Graph has %d vertices; edge community has %d vertices.
            """.format(
                graph.shape[0], edge_comm.shape[0]
            )
            raise ValueError(msg)

        siem = {
            x: {"edges": np.where(edge_comm == x), "weights": graph[edge_comm == x]}
            for x in np.unique(edge_comm)
        }
        self.model = siem
        self.K = len(self.model.keys())
        self.graph = graph
        if (self._has_been_fit):
            warnings.warn("A model has already been fit. Overwriting previous model...")
        self._has_been_fit = True
        return
    
    def summarize(self, wts, wtargs):
        """
        Allows users to compute summary statistics for each edge community in the model.
        
        Parameters
        ----------
        wts: dictionary of callables
            A dictionary of summary statistics to compute for each edge community within the model.
            The keys should be the name of the summary statistic, and each entry should be a callable
            function accepting a vector or 1-d array as the first argument. Keys are names of the summary 
            statistic, and values are the callable objects themselves.
        wtargs: dictionary of dictionaries
            A dictionary of dictionaries, where keys correspond to the names of summary statistics,
            and values are dictionaries of the named parameters desired for the summary function. The
            keys of `wts` and `wtargs` should be identical. The first key of each of the sub-dictionaries
            should have a value of `None` and should correspond to the named parameter which will be taken to be
            the edges associated with each community.
        Returns
        -------
        summary: dictionary of summary statistics
            A dictionary where keys are edge community names, and values are a dictionary of summary statistics
            associated with each community.
        """
        # check that model has been fit
        if not self._has_been_fit:
            raise UnboundLocalError("You must fit a model with fit() before summarizing the model.")
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
                raise TypeError("Each value of `wtargs` should be a sub-dictionary of class `dict`.")
            if wtarg[list(wtarg.keys())[0]] is not None:
                raise ValueError("The first entry of each sub-dictionary of wtargs should take the value None.")
                
        # equivalent of zipping the two dictionaries together
        wt_mod = {key: (wt, wtargs[key]) for key, wt in wts.items()}
        
        summary = {}
        for edge_comm in self.model.keys():
            summary[edge_comm] = {}
            for wt_name, (wt, wtarg) in wt_mod.items():
                # place the edge weights for this community in the first named
                # argument, which should be empty
                wtarg[list(wtarg.keys())[0]] = self.model[edge_comm]["weights"]
                # and store the summary statistic for this community
                summary[edge_comm][wt_name] = wt(**wtarg)
        return summary