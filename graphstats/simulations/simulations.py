import numpy as np
from graphstats.utils import import_graph, symmetrize


def er_nm(n, M, wt=None):
    """
    A function for simulating from the ER(n, M) model, a graph
    with n vertices and M edges.

    Paramaters
    ----------
        n: int
            the number of vertices
        M: int
            the number of edges, a value between
            1 and n(n-1)/2.
        wt: object
            a weight function for each of the edges, taking
            only a size argument. This weight function will
            be randomly assigned for selected edges.

    Returns
    -------
        A: array-like, shape (n, n)
    """
    if type(M) is not int:
        raise TypeError("M is not of type int.")
    if type(n) is not int:
        raise TypeError("n is not of type int.")
    if M > n*(n-1)/2:
        msg = "You have passed a number of edges, {}, exceeding n(n-1)/2, {}."
        msg = msg.format(int(M), int(n^2))
        raise ValueError(msg)
    if M < 0:
        msg = "You have passed a number of edges, {}, less than 0."
        msg = msg.format(msg)
        raise ValueError(msg)
    A = np.zeros((n, n))
    if wt is None:
        # default to binary graph
        wt = 1
    else:
        # optionally, consider weighted model
        wt = wt(size=M)
    # select M edges from upper right triangle of A, ignoring
    # diagonal, to assign connectedness
    # get triu in 1d coordinates by ravelling
    triu = np.ravel_multi_index(np.triu_indices(A.shape[0], k=1), dims=A.shape)
    # choose M of them
    idx = np.random.choice(triu, size=M, replace=False)
    # unravel back
    triu = np.unravel_index(idx, dims=A.shape)
    # assign wt function value to each of the selected edges
    np.put(A, idx, wt)
    return(symmetrize(A))
