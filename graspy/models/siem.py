import numpy as np

from ..utils import (
    import_graph,
    is_unweighted,
    remove_loops,
    symmetrize,
)
from .base import BaseGraphEstimator, _calculate_p

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
    n_components : int, optional (default=None)
        Desired dimensionality of embedding for clustering to find communities.
        ``n_components`` must be ``< min(X.shape)``. If None, then optimal dimensions 
        will be chosen by :func:`~graspy.embed.select_dimension``.
    min_comm : int, optional (default=1)
        The minimum number of communities (blocks) to consider. 
    max_comm : int, optional (default=10)
        The maximum number of communities (blocks) to consider (inclusive).
    cluster_kws : dict, optional (default={})
        Additional kwargs passed down to :class:`~graspy.cluster.GaussianCluster`
    embed_kws : dict, optional (default={})
        Additional kwargs passed down to :class:`~graspy.embed.AdjacencySpectralEmbed`
    Attributes
    ----------
    block_p_ : np.ndarray, shape (n_blocks, n_blocks)
        The block probability matrix :math:`B`, where the element :math:`B_{i, j}`
        represents the probability of an edge between block :math:`i` and block 
        :math:`j`.
    p_mat_ : np.ndarray, shape (n_verts, n_verts)
        Probability matrix :math:`P` for the fit model, from which graphs could be
        sampled.
    vertex_assignments_ : np.ndarray, shape (n_verts)
        A vector of integer labels corresponding to the predicted block that each node 
        belongs to if ``y`` was not passed during the call to ``fit``. 
    block_weights_ : np.ndarray, shape (n_blocks)
        Contains the proportion of nodes that belong to each block in the fit model.
    See also
    --------
    graspy.simulations.siem
    """
    def __init__(
        self,
        directed=True,
        loops=False
    ):
        super().__init__(directed=directed, loops=loops)
        self.model = {}

    def fit(self, graph, edge_comm, weighted=True, method='nonpar'):
        """
        Fits an SIEM to a graph.
        Parameters
        graph : array_like [nxn] or networkx.Graph with n vertices
            Input graph to fit
        edge_comm : array_like [n x n]
            A matrix giving the community assignments for each edge within the adjacency matrix
            of `graph`.

            To ignore an edge, set the value to "None"
        weighted: boolean or float (default = True)
            Boolean: True - do nothing or False - ensure everything is 0 or 1
            Float: binarize and use float as cutoff
        method: string (default = 'nonpar')
            method == 'nonpar': store all of the edge weights within a community, as a dictionary of lists
                (keys are unique community names; values are a list of the edges associated with that
                community).
            method == 'mean': store the mean of all edges associated with each community as a dictionary
                (keys are unique community names; values are the means).
            method == 'normal': store the mean, and variance, of all edges associated with each community as
            a dictionary (keys are unique community names) of dictionaries (keys are mean and variance, values are the values).
        """
        graph = import_graph(graph)

        self.n_vertices = graph.shape[0]
        if not np.ndarray.all(np.isfinite(graph)):
            raise ValueError("`graph` has non-finite entries.")
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("`graph` is not a square adjacency matrix.")
        if edge_comm.shape[0] != edge_comm.shape[1]:
            raise ValueError("`edge_comm` is not a square matrix.")
        if not np.ndarray.all(graph.shape == edge_comm.shape):
            er_msg = """
            Your edge communities do not have the same number of vertices as the graph.
            Graph has {%d} vertices; edge community has {%d} vertices.
            """.format(graph.shape[0], edge_comm.shape[0])
            raise ValueError(er_msg)
            
        if not weighted:
            if any(elem not in [0, 1] for elem in np.unique(graph)):
                msg = """You requested weighted as False, but have passed a weighted graph. 
                An unweighted graph contains only 0s or 1s. Did you mean to pass a threshold?"""
                raise ValueError(msg)
        if not isinstance(weighted, bool):
            try:
                graph[graph < weighted] = 0
                graph[graph >= weighted] = 1
            except TypeError as err:
                err.message = "You have asked for thresholding, but did not pass a number to `weighted`."
                raise
        siem = {x: {'edges': np.where(edge_comm == x), 'weights': graph[edge_comm == x]} for x in np.unique(edge_comm)}
        self.model = siem
            