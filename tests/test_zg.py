import numpy as np
from scipy.linalg import orth

from graspy.embed.svd import selectDim


def generate_data(n=10, elbows=3, seed=1):
    """
    Generate data matrix with a specific number of elbows on scree plot
    """
    np.random.seed(seed)
    x = np.random.binomial(1, .6, (n**2)).reshape(n, n)
    xorth = orth(x)
    d = np.zeros(xorth.shape[0])
    for i in range(0, len(d), int(len(d) / (elbows + 1))):
        d[:i] += 10
    A = xorth.T @ np.diag(d) @ xorth
    return A, d


def test_zg_1_elbow():
    # Should get all ones
    data, l = generate_data(10, 3)
    elbows, likelihoods, sing_vals, all_likelihoods = selectDim(data, 3)
    assert elbows[0] == 5
