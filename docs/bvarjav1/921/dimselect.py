#!/usr/bin/env python

# dimselect.py
# Created by Bijan Varjavand on 2018-09-19
# Adapted from Hayden Helm
# Email: bvarjav1@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from .svd import selectSVD #relative?


def profile_likelihood(data, n_elbows=1, threshold=0, method=selectSVD, *args, **kwargs):
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
    *args : list, optional
        Takes additional parameters
    **kwargs : dict, optional
        Takes additional keyword parameters

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
    sing_vals = method(data, *args, **kwargs)
    L = sing_vals**2
    U = L[L > threshold]
    if L.ndim == 2:
        L = np.std(U, axis=0)

    if len(U) == 0:
        msg = 'no eigenvalues ({}) greater than threshold {}'
        raise IndexError(msg.format(L, threshold))

    elbows = []
    if len(U) == 1:
        return np.array(elbows.append(U[0]))

    n = len(U)
    all_l = []
    elbow_l = []
    while len(elbows) < n_elbows and len(U) > 1:
        d = 1
        sample_var = np.var(U, ddof=1)
        sample_scale = sample_var**(1/2)
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
                elbow_l.append(likelihood)
                elbow = d
            d += 1
        if len(elbows) == 0:
            elbows.append(elbow)
        else:
            elbows.append(elbow + elbows[-1])
        U = U[elbow:]
        all_l.append(l)

    if len(elbows) == n_elbows:
        return np.array(elbows), elbow_l, U, all_l

    if len(U) == 0:
        return np.array(elbows), elbow_l, U, all_l
    else:
        elbows.append(n)
        return np.array(elbows), elbow_l, U, all_l


def gen_data(theta, n):
    """
    generates test data
    """
    top_left = np.random.binomial(1, 1/2, (int(n/2), int(n/2)))
    top_right = np.random.binomial(1, np.cos(theta)/2, (int(n/2), int(n/2)))
    top = np.concatenate((top_left, top_right), axis=1)
    bot = np.concatenate((top_right, top_left), axis=1)
    A = np.float64(np.concatenate((top, bot), axis=0))
    np.fill_diagonal(A, 0)
    return A


if __name__ == '__main__':
    data = gen_data(np.pi/2, 100)
    plt.matshow(data)
    plt.savefig('matrix2.png')
    elbows, l, sings, all_l = profile_likelihood(data, 2)
    print('elbows: ',elbows)
    print('elbow likelihoods :',l)
    print('singular values :',sings)
    print('all likelihood profiles :'.all_l.shape)
