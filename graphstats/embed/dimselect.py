#!/usr/bin/env python

# dimselect.py
# Created by Bijan Varjavand on 2018-09-19
# Adapted from Hayden Helm
# Email: bvarjav1@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import norm

def profile_likelihood(data, n_elbows = 1, threshold = 0):
    """
    Generates profile likelihood from array based on Z&G.

    Returns an array of elbows
    
    Parameters
    ----------
    data : array_like
        The matrix of data we are trying to generate profile likelihoods for.
    n_elbows : int, optional
        Number of likelihood elbows to return.

    Returns
    -------
    elbows : array_like
        Array of ZG elbows which indicate subsequent optimal embedding dimensions.
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
        msg = 'number of elbows should be an integer > 1, not {}'.format(n_elbows)
        raise ValueError(msg)

    # generate eigenvalues greater than the threshold
    pca = PCA() 
    pca.fit(data)
    L = pca.singular_values_**2
    if L.ndim == 2:
        L = np.std(U, axis = 0)
    U = L[L > threshold]
    
    if len(U) == 0:
        msg = 'no eigenvalues ({}) greater than threshold {}'.format(L, threshold)
        raise IndexError(msg)
    
    elbows = []
    if len(U) == 1:
        return np.array(elbows.append(U[0]))
    
    #U.sort() # sort
    #U = U[::-1] # reverse array so that it is sorted in descending order
    
    n = len(U)
    while len(elbows) < n_elbows and len(U) > 1:
        d = 1
        sample_var = np.var(U, ddof = 1)
        sample_scale = sample_var**(1/2)
        elbow = 0
        likelihood_elbow = 0
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
        
            if likelihood > likelihood_elbow:
                likelihood_elbow = likelihood 
                elbow = d
            d += 1
        if len(elbows) == 0:
            elbows.append(elbow)
        else:
            elbows.append(elbow + elbows[-1])
        U = U[elbow:]
        
    if len(elbows) == n_elbows:
        return np.array(elbows)
    
    if len(U) == 0:
        return np.array(elbows)
    else:
        elbows.append(n)
        return np.array(elbows) 

def gen_data(theta, n):
    top_left = np.random.binomial(1, 1/2, (int(n/2),int(n/2)))
    top_right = np.random.binomial(1, np.cos(theta)/2, (int(n/2),int(n/2)))
    top = np.concatenate((top_left,top_right), axis=1)
    bot = np.concatenate((top_right,top_left), axis=1)
    A = np.float64(np.concatenate((top,bot), axis=0))
    np.fill_diagonal(A,0)
    return A

if __name__=='__main__':
    data = gen_data(np.pi/8, 100)
    elbows = profile_likelihood(data, 2)
    print(elbows)
