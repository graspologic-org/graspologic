#!/usr/bin/env python

# svd.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds


def selectDim(X, method=TruncatedSVD, *args, **kwargs):
    """
    A function that uses likelihood profiling to determine the optimal 
    number of embedding dimensions.

    Parameters
    ----------
        X: {array-like}, shape (n_samples, n_features)
         The input data to select the optimal embedding dimensionality for.
        method: object (default TruncatedSVD)
        args: list, optional (default None)
         options taken by the desired embedding method as arguments.
        kwargs: dict, optional (default None)
         options taken by the desired embedding method as key-worded
         arguments.

    Returns
    -------
    A dictionary containing the following:
        optimal_d: {int}
         the optmial number of embedding dimensions.
        optimal_lq: {float}
         the likelihood of the optimal number of embedding dimensions.
        ds: {array-like}, shape (n_components)
         the singular values associated with the decomposition of X,
         from which optimal_d was chosen.
        lqs: {array-like}, shape (n_components)
         the likelihood profile for all possible embedding dimensions
         ds.

    See Also
    --------
        TruncatedSVD

    References:
    -----------
    Automatic dimensionality selection from the scree plot via the use of 
    profile likelihood
    Zhu, Mu and Ghodsi, Ali. CSDA 2006. 
    https://www.sciencedirect.com/science/article/pii/S0167947305002343
    """
    return {
        'optimal_d': optimal_d,
        'optimal_lq': optimal_lq,
        'ds': ds,
        'lqs': lqs
    }


def selectSVD(X, k=None):
    """
    A function for performing svd using ZG2, X = U S Vt.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        the data to perform svd on.
    k: int
        the number of dimensions to embed into. Should have
        k < min(X.shape).

    Returns
    -------
    U: array-like, shape (n_samples, k)
        the left singular vectors.
    V: array-like, shape (n_samples, k)
        the right singular vectors.
    s: array-like, shape (k)
        the singular values, as a 1d array.
    """
    if (k is None):
        selectDim(X)
    if k > min(X.shape):
        msg = "k is {}, but min(X.shape) is {}."
        msg = msg.format(k, min(X.shape))
        raise ValueError(msg)
    U, s, Vt = svds(X, k=k)
    return (U, Vt.T, s)
