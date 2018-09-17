#!/usr/bin/env python

# simulations.py
# Created by Eric Bridgeford on 2018-09-13.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
from graphstats.utils import import_graph, symmetrize


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
    if wt != 1:
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

    if wt != 1:
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")
        # optionally, consider weighted model
        wt = wt(size=M, **kwargs)

    # select uniformly btwn 0 and 1; retain edges with pchosen < p
    # get triu in 1d coordinates by ravelling
    triu = np.ravel_multi_index(idx, dims=A.shape)
    pchoice = np.random.uniform(size=len(triu))
    # connected with probability p
    triu = triu[pchoice < p]
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