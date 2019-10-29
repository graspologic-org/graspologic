import numpy as np
from graspy.simulations import sample_edges
from graspy.utils import symmetrize, cartprod
import matplotlib.pyplot as plt
import copy
import warnings

def sample_corr(P, Rho, directed=False, loops=False):
    n = np.size(P,1)
    G1 = sample_edges(P, directed = False, loops = False)
    origin_G1 = copy.deepcopy(G1)
    # prob1 = origin_G1.sum()/n**2

    P1 = copy.deepcopy(P)
    Rho = copy.deepcopy(Rho)
    for i in range(n):
        for j in range(n):
            if G1[i][j] == 1:
                P1[i][j] = P[i][j]+Rho[i][j]*(1-P[i][j])
            else:
                P1[i][j] = P[i][j]*(1-Rho[i][j])
    # prob2 = P1.sum()/n**2
    
    G2 = sample_edges(P1, directed = False, loops = False)
    G2 = G2 - np.diag(np.diag(G2))
    # prob3 = G2.sum()/(n*(n-1))
    return G1, G2, n

# run this function:
# P = 0.5 * np.ones((5,5))
# Rho = 0.3 * np.ones((5,5))
# g1,g2,n = sample_corr(P, Rho, directed=False, loops=False)
# print(g1)
# print(g2)
# print(n)