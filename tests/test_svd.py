import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

from graspy.embed.svd import selectSVD
from graspy.simulations.simulations import er_np


def test_bad_inputs():
    X = np.random.normal(size=(100, 5))
    with pytest.raises(ValueError):
        bad_algo = 'ROFLMAO'
        selectSVD(X, algorithm=bad_algo)

    with pytest.raises(ValueError):
        algorithm = 'full'
        bad_components = 1000
        selectSVD(X, n_components=bad_components, algorithm=algorithm)

    with pytest.raises(ValueError):
        algorithm = 'truncated'
        bad_components = 1000
        selectSVD(X, n_components=bad_components, algorithm=algorithm)


def test_outputs():
    np.random.seed(123)
    n = 250
    p = .25
    A = er_np(n, p)

    n_components = 3

    U_full, D_full, V_full = selectSVD(
        A, n_components=n_components, algorithm='full')
    X_full = U_full @ np.diag(np.sqrt(D_full))
    norm_full = np.linalg.norm(X_full, axis=1)

    U_trunc, D_trunc, V_trunc = selectSVD(
        A, n_components=n_components, algorithm='truncated')
    X_trunc = U_trunc @ np.diag(np.sqrt(D_trunc))
    norm_trunc = np.linalg.norm(X_trunc, axis=1)

    U_rand, D_rand, V_rand = selectSVD(
        A, n_components=n_components, algorithm='randomized', n_iter=10)
    X_rand = U_rand @ np.diag(np.sqrt(D_rand))
    norm_rand = np.linalg.norm(X_rand, axis=1)

    assert_allclose(norm_full, norm_trunc)
    assert_allclose(norm_full, norm_rand, rtol=5e-2, atol=5e-2)
