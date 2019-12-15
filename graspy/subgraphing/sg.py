import numpy as np
from scipy.stats import fisher_exact


def construct_contingency(graphs, labels):
    """
    Construct an array of edgewise contingency matrices for a set of labeled graph samples.
    
    Parameters
    ----------
    graphs: array-like, shape (n_vertices, n_vertices, s_samples)
        A set of labeled (n_vertices, n_vertices) unweighted graph samples.
        
    labels: vector, length (s_samples)
        A vector of class labels. There must be a maximum of two classes.
        
    Returns
    -------
    out: array-like, shape (n_vertices, n_vertices, 2, 2)
        An array that stores a 2-by-2 contingency matrix for each edge.
    """

    if not isinstance(graphs, np.ndarray):
        msg = "Input array 'graphs' must be np.ndarray, not {}.".format(type(graphs))
        raise TypeError(msg)
    if not isinstance(labels, (list, np.ndarray)):
        msg = "Input vector 'labels' must be list or np.ndarray, not {}.".format(
            type(labels)
        )
        raise TypeError(msg)

    shape = np.shape(graphs)
    if len(shape) != 3:
        msg = "Input array 'graphs' must be 3-dimensional with shape (n_vertices, n_vertices, s_samples)."
        raise ValueError(msg)
    if shape[0] != shape[1]:
        msg = "Input array 'graphs' must have matching number of vertices."
        raise ValueError(msg)

    if len(np.shape(labels)) != 1:
        msg = "Input vector 'labels' must be 1-dimensional."
        raise ValueError(msg)
    if len(np.unique(labels)) > 2:
        msg = "Input arrays must have a maximum of two classes, not {}.".format(
            len(np.unique(labels))
        )
        raise ValueError(msg)
    if len(labels) != shape[2]:
        msg = "Input vector length must match the number of graph samples."
        raise ValueError(msg)

    nverts = np.shape(graphs)[0]
    out = np.zeros((nverts, nverts, 2, 2))
    rowsum1 = sum(labels)
    rowsum0 = len(labels) - rowsum1
    for i in range(nverts):
        for j in range(nverts):
            a = sum(graphs[i, j, labels == 0])
            b = sum(graphs[i, j, :]) - a
            out[i, j, :, :] = [[a, rowsum0 - a], [b, rowsum1 - b]]
    return out


def estimate_signal_subgraph(graphs, labels, constraints):
    """
    Estimate the signal-subgraph of a set of labeled graph samples.
    
    The signal-subgraph estimator used depends on the dimensionality of the constraints.
    
    Parameters
    ----------
    graphs: array-like, shape (n_vertices, n_vertices, s_samples)
        A set of labeled (n_vertices, n_vertices) unweighted graph samples. If undirected, the upper or lower triangle matrices should be used.
        
    labels: vector, length (s_samples)
        A vector of class labels. There must be a maximum of two classes.
        
    constraints: int or vector
        The constraints that will be imposed onto the estimated signal-subgraph.
        
        If *constraints* is an int, *constraints* is the number of edges in the signal-subgraph.
        If *constraints* is a vector, the first element of *constraints* is the number of edges in the signal-subgraph. 
            The second element of *constraints* is the number of vertices that the signal-subgraph must be incident to.
        
    Returns
    -------
    sigsub: tuple, shape (2, n_edges)
        A tuple of a row index array and column index array, where n_edges is the size of the signal-subgraph determined by *constraints*.
    """
    tables = construct_contingency(graphs, labels)
    verts = np.shape(graphs)[0]
    sigmat = np.array(
        [
            [fisher_exact(tables[i, j, :, :])[1] for j in range(verts)]
            for i in range(verts)
        ]
    )

    if isinstance(constraints, (int, float)):  # incoherent
        nedges = constraints
        sigsub = np.dstack(
            np.unravel_index(np.argsort(sigmat.ravel()), np.shape(sigmat))
        )
        sigsub = sigsub[0, :nedges, :]
        sigsub = tuple(np.transpose(sigsub))

    elif len(constraints) == 2:  # coherent
        nedges = constraints[0]
        nverts = constraints[1]

        wset = np.unique(sigmat, axis=None)
        wcounter = 0
        wconv = 0

        while wconv == 0:
            w = wset[wcounter]
            blank = sigmat
            blank = blank > w

            score = 2 * verts - (np.sum(blank, axis=1) + np.sum(blank, axis=0))
            vscore = np.sort(score)[::-1]
            vstars = np.argsort(score)[::-1]

            if (vscore[:nverts].sum()) >= nedges:
                blank = np.ones(np.shape(sigmat))
                nstars = np.amin([len(vscore[vscore > 0]), nverts])
                vstars = vstars[:nstars]

                blank[vstars, :] = sigmat[vstars, :]
                blank[:, vstars] = sigmat[:, vstars]

                indsp = np.dstack(
                    np.unravel_index(np.argsort(blank.ravel()), np.shape(blank))
                )
                sigsub = indsp[0, :nedges, :]
                sigsub = tuple(np.transpose(sigsub))
                wconv = 1
            else:
                wcounter = wcounter + 1
                if wcounter > len(wset):
                    sigsub = []
                    wconv = 1
    else:
        msg = "Input constraints must be an int for the incoherent signal-subgraph estimator, or a vector of length 2 for the coherent subgraph estimator."
        raise TypeError(msg)
    return sigsub
