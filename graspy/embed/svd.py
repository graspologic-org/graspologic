#!/usr/bin/env python

# svd.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
import numpy as np
from scipy.stats import norm


def selectDim(data, n_elbows=1, threshold=0):
    """
    Generates profile likelihood from array based on Z&G.

    Returns an array of elbows and relevant information

    Parameters
    ----------
    data : array_like
        The matrix of data we are trying to generate profile likelihoods for.
    n_elbows : int, optional
        Number of likelihood elbows to return.
    method : object, optional
        Takes an object to calculate the svd

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
    if n_elbows < 1:
        msg = 'number of elbows should be an integer > 1, not {}'
        raise ValueError(msg.format(n_elbows))

    # generate eigenvalues greater than the threshold
    sing_vals = svds(
        data, k=min(data.shape) - 1, return_singular_vectors=False)[::-1]
    L = sing_vals**2
    L2 = L[L > threshold]
    U = L2
    if L.ndim == 2:
        L = np.std(U, axis=0)

    if len(U) == 0:
        msg = 'no eigenvalues ({}) greater than threshold {}'
        raise IndexError(msg.format(L, threshold))

    elbows = []
    if len(U) == 1:
        return np.array(elbows.append(U[0])) + 1

    n = len(U)
    all_l = []
    elbow_l = []
    while len(elbows) < n_elbows and len(U) > 1:
        d = 1
        sample_var = np.var(U, ddof=1)
        sample_scale = sample_var**(1 / 2)
        elbow = 0
        likelihood_elbow = 0
        l = []
        while d < len(U):
            mean_sig = np.mean(U[:d])
            mean_noise = np.mean(U[d:])
            sig_likelihood = 0
            noise_likelihood = 0
            for i in range(d):
                sig_likelihood += norm.pdf(U[i], mean_sig, sample_scale)
            for i in range(d, len(U)):
                noise_likelihood += norm.pdf(U[i], mean_noise, sample_scale)

            likelihood = noise_likelihood + sig_likelihood
            l.append(likelihood)

            if likelihood > likelihood_elbow:
                likelihood_elbow = likelihood
                elbow = d
            d += 1
        elbow_l.append(likelihood_elbow)
        if len(elbows) == 0:
            elbows.append(elbow)
        else:
            elbows.append(elbow + elbows[-1])
        U = U[elbow:]
        all_l.append(l)

    if len(elbows) == n_elbows:
        return np.array(elbows) + 1, elbow_l, L2, all_l

    if len(U) == 0:
        return np.array(elbows) + 1, elbow_l, L2, all_l
    else:
        elbows.append(n)
        return np.array(elbows) + 1, elbow_l, L2, all_l

    return {
        'optimal_d': np.array(elbows) + 1,
        'optimal_lq': elbow_l,
        'ds': L2,
        'lqs': all_l
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
    if k > min(
            X.shape
    ):  #TODO this method does not properly catch error if k=min(X.shape),
        # also may be unecessary (see svds error catching)
        msg = "k is {}, but min(X.shape) is {}."
        msg = msg.format(k, min(X.shape))
        raise ValueError(msg)
    U, s, Vt = svds(X, k=k)
    return (U, Vt.T, s)
