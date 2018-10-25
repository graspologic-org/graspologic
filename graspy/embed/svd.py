#!/usr/bin/env python

# svd.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
import numpy as np
from scipy.stats import norm


def _compute_likelihood(arr):
    """
    """
    n_elements = len(arr)
    likelihoods = np.zeros(n_elements)

    for idx in range(1, n_elements + 1):
        # split into two samples
        s1 = arr[:idx]
        s2 = arr[idx:]
        if s2.size == 0:  # deal with when idx == n_elements
            s2 = np.zeros(1)

        # compute means
        mu1 = np.mean(s1)
        mu2 = np.mean(s2)

        # compute pooled variance
        variance = ((np.sum((s1 - mu1)**2) + np.sum(
            (s2 - mu2)**2))) / (n_elements - 2)
        std = np.sqrt(variance)

        # compute log likelihoods
        likelihoods[idx - 1] = np.sum(norm.logpdf(
            s1, loc=mu1, scale=std)) + np.sum(
                norm.logpdf(s2, loc=mu2, scale=std))

    return likelihoods


def selectDim(X, n_components=None, n_elbows=2, threshold=None):
    """
    Generates profile likelihood from array based on Z&G.

    Returns an array of elbows and relevant information

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input array generate profile likelihoods for.
    n_components : int, optional, default: None.
        Number of components to embed. If None, ``n_components = 
        floor(log2(min(n_samples, n_features)))``.
    n_elbows : int, optional, default: 1.
        Number of likelihood elbows to return. Must be > 1. 
    threshold : float, int, optional, default: None
        If given, only consider the singular values that are > threshold. Must
        be >= 0.

    Returns
    -------
    elbows : array_like
        ZG elbows which indicate subsequent optimal embedding dimensions.
    likelihoods : array_like
        Array of likelihoods of the optimal embedding dimensions.
    sing_vals : array_like
        The singular values of the data array post-threshold.
    all_likelihoods : array_like
        The likelihood profiles of all embedding dimensions.

    Other Parameters
    ----------------
    threshold : float, optional
        Ignores eigenvalues smaller than this.

    Raises
    ------
    ValueError
        If n_elbows is :math:`< 1`.

    References
    ----------
    .. [1] Zhu, M. and Ghodsi, A. (2006).
        Automatic dimensionality selection from the scree plot via the use of profile likelihood.
        Computational Statistics & Data Analysis, 51(2), pp.918-930.

    """
    # Handle n_elbows
    if not isinstance(n_elbows, int):
        msg = 'n_elbows must be an integer, not {}.'.type(n_elbows)
        raise ValueError(msg)
    elif n_elbows < 1:
        msg = 'number of elbows should be an integer > 1, not {}.'.format(
            n_elbows)
        raise ValueError(msg)

    # Handle threshold
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            msg = 'threshold must be an integer or a float, not {}.'.type(
                threshold)
            raise ValueError(msg)
        elif threshold < 0:
            msg = 'threshold must be >= 0, not {}.'.format(threshold)
            raise ValueError(msg)

    # Handle input data
    if not isinstance(X, np.ndarray):
        msg = 'X must be a numpy array, not {}.'.format(type(X))
        raise ValueError(msg)

    if X.ndim > 2:
        msg = 'X must be a 1d or 2d-array, not {}d array.'.format(X.ndim)
        raise ValueError(msg)
    elif np.min(X.shape) == 1:
        msg = 'X must have more than 1 samples or 1 features.'
        raise ValueError(msg)

    # Handle max components
    if n_components is None:
        if np.min(X.shape) == 1:
            k = 1
        else:
            k = np.floor(np.log2(np.min(X.shape)))
    elif not isinstance(n_components, int):
        msg = 'n_components must be an integer, not {}.'.format(
            type(n_components))
    else:
        k = n_components

    # Singular values in decreasing order
    D = svds(A=X, k=k, return_singular_vectors=False)[::-1]

    #L = sing_vals**2
    if threshold is not None:
        D = D[D > threshold]

    if len(U) == 0:
        msg = 'No singular values greater than threshold {}.'
        raise IndexError(msg.format(threshold))
    elif len(U) <= n_elbows:
        msg = 'n_elbows must between {}, the number of thresholded \
        singular values'.format(len(U))

    if len(U) == 1:
        return np.array(elbows.append(U[0])) + 1

    idx = 0
    elbows = []
    for _ in range(n_elbows):
        likelihoods = _compute_likelihood(D[idx:])
        idx += np.argmax(liklihoods)
        elbows.append(idx + 1)

    if len(elbows) == n_elbows:
        return elbows, elbow_l, U, all_l

    if len(U) == 0:
        return np.array(elbows) + 1, elbow_l, U, all_l
    else:
        elbows.append(n)
        return np.array(elbows) + 1, elbow_l, U, all_l


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
    if k > min(
            X.shape
    ):  #TODO this method does not properly catch error if k=min(X.shape),
        # also may be unecessary (see svds error catching)
        msg = "k is {}, but min(X.shape) is {}."
        msg = msg.format(k, min(X.shape))
        raise ValueError(msg)
    U, s, Vt = svds(X, k=k)
    return (U, Vt.T, s)
