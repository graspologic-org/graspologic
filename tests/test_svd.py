import pytest
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy import array_equal, allclose
from numpy.linalg import norm

from graphstats.embed.svd import selectDim
from scipy.linalg import orth


def generate_data(n=10, elbows=3, seed=1):
    """
    Generate data matrix with a specific number of elbows on scree plot
    """
    np.random.seed(seed)
    x = np.random.binomial(1,.6,(n**2)).reshape(n,n)
    xorth = orth(x)
    d = np.zeros(xorth.shape[0])
    #for i in range(0,len(d), int(len(d)/(elbows+1))):
    d[:5] += 10
    A = xorth.T @ np.diag(d) @ xorth
    return A,d

def test_zg_1_elbow():
    # Should get all ones
    data,l = generate_data(10,1)
    elbows, likelihoods, sing_vals, all_likelihoods = selectDim(data, 3)
    assert elbows[0]==6

 
