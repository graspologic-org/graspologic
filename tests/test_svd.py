import pytest
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy import array_equal, allclose
from numpy.linalg import norm

from graphstats.embed.svd import selectDim


def generate_data(n, elbows, seed=1):
    """
    Generate data matrix with a specific number of elbows on scree plot

    Parameters
    ----------
    n : int
        Dimension of the matrix.
    elbows : int
        Number of elbows in scree plot.

    Returns
    -------
    data : array_like
        n by n matrix, which has some scree plot elbows
    """
    np.random.seed(seed)
    X = np.random.normal(0,1,n)
    l = np.zeros(n)
    for i in range(0,n,int(n/elbows)):
        l[:i]+=10
    l = l.reshape(-1,1)
    # Generate matrix
    data = np.dot(X, X.T)
    return data

def test_zg_1_elbow():
    # Should get all ones
    data = generate_data(100,1)
    elbows, likelihoods, sing_vals, all_likelihoods = selectDim(data, 3)
    plt.plot(sing_vals)
    plt.plot(elbows, sing_vals[elbows], 'ro')
    plt.show()
    assert True

if __name__=='__main__':
    test_zg_1_elbow()

def test_omni_matrix_random():
    expected_output = np.array(
        [[0., 0., 1., 0., 0., 0.5], [0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0.5, 0., 0.], [0., 0., 0.5, 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.], [0.5, 0., 0., 0., 0., 0.]])

    np.random.seed(2)
    graphs = [er_np(3, .3) for _ in range(2)]

    A = _get_omni_matrix(graphs)
    assert array_equal(A, expected_output)


def test_omni_matrix_invalid_inputs():
    with pytest.raises(ValueError):
        empty_list = []
        omni = OmnibusEmbed(k=2)
        omni.fit(empty_list)

    with pytest.raises(ValueError):
        wrong_shapes = [np.zeros((10, 10)), np.zeros((20, 20))]
        omni = OmnibusEmbed(k=2)
        omni.fit(wrong_shapes)

    with pytest.raises(TypeError):
        wrong_dtypes = [1, 2, 3]
        omni = OmnibusEmbed(k=2)
        omni.fit(wrong_dtypes)


def test_omni_matrix_symmetric():
    np.random.seed(3)
    n = 15
    p = 0.4

    n_graphs = [2, 5, 100]
    for n in n_graphs:
        graphs = [er_np(n, p) for _ in range(n)]
        output = _get_omni_matrix(graphs)
        assert is_symmetric(output)


def test_omni_embed():
    """
    We compare the difference of norms of OmniBar and ABar.
    ABar is the lowest variance estimate of the latent positions X.
    OmniBar should be reasonablly close to ABar when n_vertices is high. 
    """
    def compute_bar(arr):
        n = arr.shape[0] // 2
        return (arr[:n] + arr[n:]) / 2

    X, A1, A2 = generate_data(1000)
    Abar = (A1 + A2) / 2

    np.random.seed(11)
    omni = OmnibusEmbed(k=3)
    OmniBar = compute_bar(omni.fit_transform([A1, A2]))
    
    omni = OmnibusEmbed(k=3)
    ABar = compute_bar(omni.fit_transform([Abar, Abar]))

    tol = 1.e-2
    assert allclose(norm(OmniBar, axis=1), norm(ABar, axis=1), rtol=tol, atol=tol)
