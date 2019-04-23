import unittest
import pytest
import graspy as gs
import numpy as np
from graspy.embed.oosase import OutOfSampleAdjacencySpectralEmbed
from graspy.simulations.simulations import sbm


def test_oosase_predict():
    np.random.seed(8888)

    P = np.array([[0.8, 0.2], [0.2, 0.8]])
    n = 200
    verts_per_community = [100, 100]
    A = sbm(verts_per_community, P)

    ase_object = OutOfSampleAdjacencySpectralEmbed()

    in_sample_A = A[: int(np.ceil(0.8 * n)), : int(np.ceil(0.8 * n))]
    X = A[int(np.ceil(0.8 * n)) :, : int(np.ceil(0.8 * n))]

    ase_object.fit(in_sample_A)

    # X is an array
    with pytest.raises(TypeError):
        X_hat = ase_object.predict(0)

    # len(X) > 1000
    with pytest.raises(ValueError):
        X_hat = ase_object.predict(np.zeros(1000))

    # X.shape[0] != n and X.shape[1] != n
    with pytest.raises(ValueError):
        X_hat = ase_object.predict(np.ones((1000, 1000)))

    # no tensors
    with pytest.raises(ValueError):
        X_hat = ase_object.predict(np.ones((1, 1, 1)))

    # no zero rows 1-d
    with pytest.raises(ValueError):
        X_hat = ase_object.predict(np.zeros(int(np.ceil(0.8 * n))))

    # no zero rows 2-d
    with pytest.raises(ValueError):
        X_hat = ase_object.predict(np.zeros((1000, int(np.ceil(0.8 * n)))))


def test_fit_predict():
    np.random.seed(8888)

    P = np.array([[0.8, 0.2], [0.2, 0.8]])
    n = 200
    verts_per_community = [100, 100]
    A = sbm(verts_per_community, P)

    ase_object1 = OutOfSampleAdjacencySpectralEmbed(
        semi_supervised=True, in_sample_idx=range(int(np.ceil(0.8 * n)))
    )
    ase_object1.fit(A)
    oos_A = A[int(np.ceil(0.8 * n)) :, : int(np.ceil(0.8 * n))]
    oos_embed = ase_object1.predict(oos_A)
    X_hat1 = ase_object1.latent_left_

    np.random.seed(8888)
    ase_object2 = OutOfSampleAdjacencySpectralEmbed(
        semi_supervised=True, in_sample_idx=range(int(np.ceil(0.8 * n)))
    )
    X_hat2 = ase_object2.fit_predict(A)

    assert np.sum(X_hat2 == X_hat1) == np.prod(X_hat2.shape)
