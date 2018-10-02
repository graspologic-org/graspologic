#!/usr/bin/env python

# dimselect.py
# Copyright (c) 2017. All rights reserved.

import numpy as np
from scipy.stats import norm

def profile_likelihood(L, n_elbows = 1, threshold = 0):
    """
    An implementation of profile likelihood as outlined in Zhu and Ghodsi.
    
    Inputs
        L - An ordered or unordered list of eigenvalues
        n - The number of elbows to return
        threshold - Smallest value to consider. Nonzero thresholds will affect elbow selection.

    Return
        elbows - A numpy array containing elbows


    """

    U = L.copy()

    if type(U) == list: # cast to array for functionality later
        U = np.array(U)
    
    if n_elbows == 0: # nothing to do..
        return np.array([])
    
    if U.ndim == 2:
        U = np.std(U, axis = 0)
    
    # select values greater than the threshold
    U = U[U > threshold]
    
    if len(U) == 0:
        return np.array([])
    
    elbows = []
    
    if len(U) == 1:
        return np.array(elbows.append(U[0]))
    
    U.sort() # sort
    U = U[::-1] # reverse array so that it is sorted in descending order
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
