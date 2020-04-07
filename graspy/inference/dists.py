import numpy as np
from scipy.spatial.distance import cdist

def euclidean(x):
    """Default euclidean distance function calculation"""
    return cdist(x, x, metric="euclidean")

def gaussian(x):
    """Default medial gaussian kernel similarity calculation"""
    l1 = cdist(x, x, "cityblock")
    mask = np.ones(l1.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    gamma = 1.0 / (2 * (np.median(l1[mask]) ** 2))
    K = np.exp(-gamma * cdist(x, x, "sqeuclidean"))
    return 1 - K/np.max(K)
