#!/usr/bin/env python

# simulations.py
# Created by Eric Bridgeford on 2018-09-13.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
from graphstats.utils import import_graph, symmetrize


def cartprod(*arrays):
     N = len(arrays)
     return np.transpose(np.meshgrid(*arrays, indexing='ij'), 
                      np.roll(np.arange(N + 1), -1)).reshape(-1, N)

def weighted_sbm(n, P, Wt=1, directed=False, loops=False, Wtargs=None):
    """
    A function for simulating from a weighted sbm.

    Paramaters
    ----------
        n: list of int, shape (n_communities)
            the number of vertices in each community. Communities
            are assigned n[0], n[1], ...
        P: array-like, shape (n_communities, n_communities)
            the probability of an edge between each of the communities,
            where P[i, j] indicates the probability of a connection
            between edges in communities [i, j]. 0 < P[i, j] < 1
            for all i, j.
        directed: boolean
            whether or not the graph will be directed.
        loops: boolean
            whether to allow self-loops for vertices.
        Wt: object or array-like, shape (n_communities, n_communities)
            + if Wt is an object, a weight function to use globally over
            the sbm for assigning weights. 1 indicates to produce a binary
            graph.
            + if Wt is an array-like, a weight function for each of
            the edge communities. Wt[i, j] corresponds to the weight function
            between communities i and j. If the entry is a function, should
            accept an argument for size. An entry of Wt[i, j] = 1 will produce a
            binary subgraph over the i, j community.
        Wtargs: dictionary or array-like, shape (n_communities, n_communities)
            + if Wt is an object, Wtargs corresponds to the trailing arguments
            to pass to the weight function.
            + if Wt is an array-like, Wtargs[i, j] corresponds to trailing
            arguments to pass to Wt[i, j].            

    Returns
    -------
        A: array-like, shape (n, n)
            the adjacency matrix.
    """
    # type checking
    if type(n) is not list:
        raise TypeError("n is not a list.")
    if not all(type(i) == int for i in n):
        raise TypeError("An entry of n is not an integer.")
    if type(directed) is not bool:
        raise TypeError("directed is not of type bool.")
    if type(loops) is not bool:
        raise TypeError("loops is not of type bool.")

    K = len(n)  # the number of communities
    counter = 0
    # get a list of community indices
    cmties = []
    for i in range(0, K):
        cmties.append(range(counter, counter + n[i]))
        counter += n[i]
    if type(P) != np.ndarray:
        raise TypeError("P is not an ndarray.")
    if P.shape != (K, K):
        er_msg = "P is not a square, len(n) x len(n) matrix. P of shape {}"
        er_msg = er_msg.format(P.shape)
        raise ValueError(er_msg)
    if (not directed) and np.any(P != P.T):
        raise ValueError("Specified undirected, but P is directed.")
    if (not directed) and isinstance(Wt, np.ndarray):
        if np.any(Wt != Wt.T):
            raise ValueError("Specified undirected, but Wt is directed.")
    if (not directed) and isinstance(Wtargs, np.ndarray):
        if np.any(Wtargs != Wtargs.T):
            raise ValueError("Specified undirected, but Wtargs is directed.")
    if type(Wt) == np.ndarray:
        if Wt.shape != (K, K):
            er_msg = "Wt is not a square, len(n) x len(n) matrix."
            er_msg += " Wt is of shape {}".format(Wt.shape)
            raise ValueError(er_msg)
        if Wt.shape != Wtargs.shape:
            er_msg = "Wt is of shape {}, but Wtargs is of shape {}."
            er_msg += " They should have the same shape."
            raise ValueError(er_msg) 
    else:
        # reshape to make an ndarray for each community
        Wt = np.full(P.shape, Wt, dtype=object)
        Wtargs = np.full(P.shape, Wtargs, dtype=object)
    # check probabilities are valid
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("Your probability matrix is not 0 < P[i, j] < 1.")

    A = np.zeros((sum(n), sum(n)))

    for i in range(0, K):
        if directed:
            jrange = range(0, K)
        else:
            jrange = range(i, K)
        for j in jrange:
            wt = Wt[i, j]; wtargs = Wtargs[i, j]; p = P[i, j]
            # identify submatrix for community i, j
            # cartesian product to identify edges for community i,j pair
            cprod = cartprod(cmties[i], cmties[j])
            # get idx in 1d coordinates by ravelling
            triu = np.ravel_multi_index((cprod[:,0], cprod[:,1]), dims=A.shape)
            pchoice = np.random.uniform(size=len(triu))
            # connected with probability p
            triu = triu[pchoice < p]
            if type(wt) is not int:
                if not callable(wt):
                    raise TypeError("You have not passed a function for wt.")
                wt = wt(size=len(triu), **wtargs)
            triu = np.unravel_index(triu, dims=A.shape)
            A[triu] = wt
    if not loops:
        A = A - np.diag(np.diag(A))
    if not directed:
        A = symmetrize(A)
    return(A)
    if not loops:
        A = remove_loops(A)
    if not directed:
        A = symmetrize(A)
    return(A)

def binary_sbm(n, P, directed=False, loops=False):
    """
    A function for simulating from a simple sbm.

    Paramaters
    ----------
        n: list of int, shape (n_communities)
            the number of vertices in each community. Communities
            are assigned n[0], n[1], ...
        P: array-like, shape (n_communities, n_communities)
            the probability of an edge between each of the communities,
            where P[i, j] indicates the probability of a connection
            between edges in communities [i, j]. 0 < P[i, j] < 1
            for all i, j.
        directed: boolean
            whether or not the graph will be directed.
        loops: boolean
            whether to allow self-loops for vertices.
            + if Wt is an array-like, a weight function for each of
            the edge communities. Wt[i, j] corresponds to the weight function
            between communities i and j. If the entry is a function, should
            accept an argument for size. An entry of Wt[i, j] = 1 will produce a
            binary subgraph over the i, j community.

    Returns
    -------
        A: array-like, shape (n, n)
            the adjacency matrix.
    """
    return(weighted_sbm(n, P, directed=directed, loops=loops))

def zi_nm(n, M, wt=1, directed=False, loops=False, **kwargs):
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
        directed: boolean
            whether or not the graph will be directed.
        loops: boolean
            whether to allow self-loops for vertices.
        wt: object
            a weight function for each of the edges, taking
            only a size argument. This weight function will
            be randomly assigned for selected edges. If 1,
            graph produced is binary.
        kwargs: dictionary
            optional arguments for parameters that can be passed
            to weight function wt.

    Returns
    -------
        A: array-like, shape (n, n)
            the adjacency matrix.
    """
    if type(M) is not int:
        raise TypeError("M is not of type int.")
    if type(n) is not int:
        raise TypeError("n is not of type int.")
    if type(directed) is not bool:
        raise TypeError("directed is not of type bool.")
    if type(loops) is not bool:
        raise TypeError("loops is not of type bool.")
    # check for loopiness
    if loops:
        er_msg = "n^2"
        Mmax = n**2
    else:
        # get all indices including diagonal
        er_msg = "n(n-1)"
        Mmax = n*(n-1)

    A = np.zeros((n, n))
    # check if directedness is desired
    if directed:
        if loops:
            # use all of the indices
            idx = np.where(np.logical_not(A))
        else:
            # use only the off-diagonal indices
            idx = np.where(~np.eye(n, dtype=bool))
    else:
        # use upper-triangle indices, and ignore diagonal according
        # to loops argument
        idx = np.triu_indices(n, k=int(loops == False))
        er_msg += "/2"
        Mmax = Mmax/2

    # check whether M exceeds the maximum possible M
    if M > Mmax:
        msg = "You have passed a number of edges, {}, exceeding {}, {}."
        msg = msg.format(int(M), er_msg, Mmax)
        raise ValueError(msg)
    if M < 0:
        msg = "You have passed a number of edges, {}, less than 0."
        msg = msg.format(msg)
        raise ValueError(msg)

    # check weight function
    if type(wt) is not int:
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")
        # optionally, consider weighted model
        wt = wt(size=M, **kwargs)

    # get idx in 1d coordinates by ravelling
    triu = np.ravel_multi_index(idx, dims=A.shape)
    # choose M of them
    triu = np.random.choice(triu, size=M, replace=False)
    # unravel back
    triu = np.unravel_index(triu, dims=A.shape)
    # assign wt function value to each of the selected edges
    A[triu] = wt
    if not directed:
        A = symmetrize(A)
    return(A)

def zi_np(n, p, wt=1, directed=False, loops=False, **kwargs):
    """
    A function for simulating from the zi(n, M, params) model, a graph
    with n vertices and M edges and edge-weight function wt.

    Paramaters
    ----------
        n: int
            the number of vertices
        p: float
            the probability of an edge existing between two vertices,
            between 0 and 1.
        directed: boolean
            whether or not the graph will be directed.
        loops: boolean
            whether to allow self-loops for vertices.
        wt: object
            a weight function for each of the edges, taking
            only a size argument. This weight function will
            be randomly assigned for selected edges. If 1,
            graph produced is binary.
        kwargs: dictionary
            optional arguments for parameters that can be passed
            to weight function wt.

    Returns
    -------
        A: array-like, shape (n, n)
            the adjacency matrix.
    """
    # type checking
    if type(p) is not float:
        raise TypeError("p is not of type float.")
    if type(n) is not int:
        raise TypeError("n is not of type int.")
    if type(directed) is not bool:
        raise TypeError("directed is not of type bool.")
    if type(loops) is not bool:
        raise TypeError("loops is not of type bool.")
    # check p
    if p < 0:
        msg = "You have passed a probability, {}, less than 0."
        msg = msg.format(float(p))
        raise ValueError(msg)
    if p > 1:
        msg = "You have passed a probability, {}, greater than 1."
        msg = msg.format(float(p))
        raise ValueError(msg)

    # check for loopiness
    if loops:
        er_msg = "n^2"
        Mmax = n**2
    else:
        # get all indices including diagonal
        er_msg = "n(n-1)"
        Mmax = n*(n-1)

    A = np.zeros((n, n))
    # check if directedness is desired
    if directed:
        if loops:
            # use all of the indices
            idx = np.where(np.logical_not(A))
        else:
            # use only the off-diagonal indices
            idx = np.where(~np.eye(n, dtype=bool))
    else:
        # use upper-triangle indices, and ignore diagonal according
        # to loops argument
        idx = np.triu_indices(n, k=int(loops == False))
        er_msg += "/2"
        Mmax = Mmax/2

    # select uniformly btwn 0 and 1; retain edges with pchosen < p
    # get triu in 1d coordinates by ravelling
    triu = np.ravel_multi_index(idx, dims=A.shape)
    pchoice = np.random.uniform(size=len(triu))
    # connected with probability p
    triu = triu[pchoice < p]

    if type(wt) is not int:
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")
        # optionally, consider weighted model
        wt = wt(size=len(triu), **kwargs)

    # unravel back
    triu = np.unravel_index(triu, dims=A.shape)
    # assign wt function value to each of the selected edges
    A[triu] = wt
    if not directed:
        A = symmetrize(A)
    return(A)

def er_nm(n, M):
    """
    A function for simulating from the ER(n, M) model, a simple graph
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
    return(zi_nm(n, M, wt=1))

def er_np(n, p):
    """
    A function for simulating from the ER(n, p) model, a simple graph
    with n vertices and a probability p of edges being connected.

    Paramaters
    ----------
        n: int
            the number of vertices
        p: float
            the probability of an edge existing between two vertices,
            between 0 and 1.

    Returns
    -------
        A: array-like, shape (n, n)
    """
    return(zi_np(n, p, wt=1))