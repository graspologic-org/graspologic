import numpy as np

from ..types import AdjacencyMatrix


def compute_density(adjacency: AdjacencyMatrix, loops: bool = False) -> float:
    """
    For a given graph, this function computes the graph density, defined as the actual number of edges divided by the total possible number
    of edges in the graph.

    Parameters
    ----------
    adjacency: int, array shape (n_nodes,n_nodes)
        The adjancy matrix for the graph. Edges are denoted by 1s while non-edges are denoted by 0s.

    loops: boolean
        Optional variable to select whether to include self-loops (i.e. connections between a node and itself). Default is "false," meaning
        such connections are ignored.

    Returns
    -------
    n_edges/n_possible: float
        The computed density, calculated as the total number of edges divided by the total number of possible edges.

    """
    n_edges = np.count_nonzero(adjacency)
    n_nodes = adjacency.shape[0]
    n_possible = n_nodes**2
    if not loops:
        n_possible -= n_nodes
    return n_edges / n_possible


def compute_density_adjustment(
    adjacency1: AdjacencyMatrix, adjacency2: AdjacencyMatrix
) -> float:
    """
    Computes the density adjustment to be used when testing the hypothesis that the density of one network is equal to a fixed parameter
    times the density of a second network. This function first calls the compute_density function above to compute the densities of both
    networks, then computes an odds ratio by calculating the odds of an edge in each network and taking the ratio of the results.

    Parameters
    ----------
    adjacency1: int, array of size (n_nodes1,n_nodes1)
        Adjacency matrix for the first graph. 1s represent edges while 0s represent the absence of an edge. The array is a square of side length
        n_nodes1, where this corresponds to the number of nodes in graph 1.

    adjacency2: int, array of size (n_nodes2,n_nodes2)
        Same as above, but for the second graph.

    Returns
    ---------
    odds_ratio: float
        Computed as the ratio of the odds of an edge in graph 1 to the odds of an edge in graph 2.

    """
    density1 = compute_density(adjacency1)
    density2 = compute_density(adjacency2)
    # return density1 / density2
    odds1 = density1 / (1 - density1)
    odds2 = density2 / (1 - density2)
    odds_ratio = odds1 / odds2
    return odds_ratio
