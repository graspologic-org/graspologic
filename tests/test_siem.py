import unittest
from graspy.simulations import siem
from graspy.models import SIEMEstimator
import numpy as np


def modular_edges(n):
    """
    A function for generating modular sbm edge communities.
    """
    m = int(n/2)
    edge_comm = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if ( (i<m) & (j<m)) or ( (i>=m ) & (j>=m) ):
                edge_comm[i,j] = 1
            else:
                edge_comm[i,j] = 2
    return edge_comm

def nuis_edges(n):
    """
    A function for generating doubly modular sbm.
    """
    m = int(n/2)
    m4 = int(7*n/8)
    m3 = int(5*n/8)
    m2 = int(3*n/8)
    m1 = int(1*n/8)
    edge_comm = [[],[],[]]
    for i in range(n):
        for j in range(n):
            if ( (i<m) & (j<m)) or ( (i>=m ) & (j>=m) ):
                edge_comm[i,j] = 1
            elif (((i >= m3) & (i <= m4)) & ((j >= m1) & (j <= m2))) or (((i >= m1) & (i <= m2)) & ((j >= m3) & (j <= m4))):
                edge_comm[i,j] = 3
            else:
                edge_comm[i,j] = 2
    return edge_comm
    

def diag_edges(n):
    """
    A function for generating diagonal SIEM edge communities.
    """
    m = int(n/2)
    edge_comm = [[],[]]
    for i in range(n):
        for j in range(n):
            if (i == j + m) or (j == i + m):
                edge_comm[i,j] = 1
            else:
                edge_comm[i,j] = 2
    return edge_comm

class 