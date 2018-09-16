import numpy as np
from graphstats.utils import import_graph, symmetrize


def zi_nm(n, M, wt=1, **kwargs):
    """
    A function for simulating from the zi(n, M, params) model, a graph
    with n vertices and M edges and edge-weight function wt.

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
        kwargs: dictionary
            optional arguments for parameters that can be passed
            to weight function wt.

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
    if wt != 1:
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")
        # optionally, consider weighted model
        wt = wt(size=M, **kwargs)
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


def zi_np(n, M, wt='ER', **kwargs):
    """
    A function for simulating from the zi(n, M, params) model, a graph
    with n vertices and M edges and edge-weight function wt.

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
        kwargs: dictionary
            optional arguments for parameters that can be passed
            to weight function wt.

    Returns
    -------
        A: array-like, shape (n, n)
    """
    if type(p) is not float:
        raise TypeError("p is not of type float.")
    if type(n) is not int:
        raise TypeError("n is not of type int.")
    if p < 0:
        msg = "You have passed a probability, {}, less than 0."
        msg = msg.format(float(p))
        raise ValueError(msg)
    if p > 1:
        msg = "You have passed a probability, {}, greater than 1."
        msg = msg.format(float(p))
        raise ValueError(msg)
    A = np.zeros((n, n))
    # select uniformly btwn 0 and 1; retain edges with pchosen < p
    # get triu in 1d coordinates by ravelling
    triu = np.ravel_multi_index(np.triu_indices(A.shape[0], k=1), dims=A.shape)
    pchoice = np.random.uniform(size=len(triu))
    # connected with probability p
    triu = triu[pchoice < p]
    # unravel back
    if wt != 'ER':
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")
        # optionally, consider weighted model
        wt = wt(size=len(triu), **kwargs)
    triu = np.unravel_index(idx, dims=A.shape)
    # assign wt function value to each of the selected edges
    np.put(A, idx, wt)
    return(symmetrize(A))


def er_nm(n, M):
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
    return(zi_nm(n, M, wt='ER'))

def er_np(n, M, wt=None):
    """
    A function for simulating from the ER(n, p) model, a graph
    with n vertices and a probability p of edges being connected.

    Paramaters
    ----------
        n: int
            the number of vertices
        p: float
            the probability of an edge existing between two vertices,
            between 0 and 1.
        wt: object
            a weight function for each of the edges, taking
            only a size argument. This weight function will
            be randomly assigned for selected edges.

    Returns
    -------
        A: array-like, shape (n, n)
    """
    return(zi_np(n, p, wt='ER'))